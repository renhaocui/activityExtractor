import json, re
import numpy as np
from keras.layers import Dense, LSTM, Conv1D, Flatten, Reshape, Bidirectional, Input, MaxPooling1D, dot, add, GRU, Lambda
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from utilities import word2vecReader, evaluation
from wordsegment import load
import sys
from sklearn.model_selection import StratifiedKFold
from keras_self_attention import SeqSelfAttention
reload(sys)
sys.setdefaultencoding('utf8')

vocabSize = 10000
tweetLength = 25
yelpLength = 100
embeddingVectorLength = 200
batch_size = 100

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}
POSMapper = {'N': 'N', 'O': 'N', '^': 'N', 'S': 'N', 'Z': 'N', 'L': 'N', 'M': 'N',
             'V': 'V', 'A': 'A', 'R': 'R', '@': '@', '#': '#', '~': '~', 'E': 'E', ',': ',', 'U': 'U',
             '!': '0', 'D': '0', 'P': '0', '&': '0', 'T': '0', 'X': '0', 'Y': '0', '$': '0', 'G': '0'}
load()

def removeLinks(input):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, '')
    return input


def hourMapper(hour):
    input = int(hour)
    if 0 <= input < 6:
        output = 0
    elif 6 <= input < 12:
        output = 1
    elif 12 <= input < 18:
        output = 2
    else:
        output = 3
    return output


def overSample(inputList, ratio):
    if ratio == 1.0:
        return []
    outputList = []
    size = len(inputList)
    for i in range(int(size*(ratio-1))):
        outputList.append(inputList[i%size])
    return outputList


def extractPOS(inputList, mode='all', breakEmoji=True):
    posOutput = ''
    contentOutput = ''
    for item in inputList:
        if breakEmoji:
            emojis1 = re.findall(r'\\u....', item[0].encode('unicode-escape'))
            emojis2 = re.findall(r'\\U........', item[0].encode('unicode-escape'))
            emojis = emojis1 + emojis2
            if len(emojis) > 0:
                for emoji in emojis:
                    contentOutput += emoji + ' '
                    posOutput += 'E' + ' '
            else:
                contentOutput += item[0] + ' '
                if mode == 'all':
                    posOutput += item[1] + ' '
                else:
                    posOutput += POSMapper[item[1]] + ' '
        else:
            contentOutput += item[0] + ' '
            if mode == 'all':
                posOutput += item[1] + ' '
            else:
                posOutput += POSMapper[item[1]] + ' '
        if len(contentOutput.split(' ')) != len(posOutput.split(' ')):
            print('error')
            print(contentOutput)
    return contentOutput.lower().strip().encode('utf-8'), posOutput.strip().encode('utf-8')


def loadYelpData(modelName, char, embedding):
    print('Loading...')
    if 'yelp' in modelName:
        textLength = yelpLength
    else:
        textLength = tweetLength
    contents_train = []
    contents_val = []
    labels_train = []
    labels_val = []
    ids_train = []
    ids_val = []
    trainFile = open('data/yelp/consolidateData_' + modelName + '_train.json', 'r')
    valFile = open('data/yelp/consolidateData_' + modelName + '_test.json', 'r')

    for line in trainFile:
        data = json.loads(line.strip())
        histLen = len(data['reviews'])
        contents_train.append(data['reviews'][histLen-1]['text'].lower().encode('utf-8'))
        labels_train.append(data['reviews'][histLen-1]['stars'])
        ids_train.append(str(data['reviews'][histLen-1]['review_id']))
    ids_train = np.array(ids_train)
    labels_train = np.array(labels_train)

    for line in valFile:
        data = json.loads(line.strip())
        histLen = len(data['reviews'])
        contents_val.append(data['reviews'][histLen-1]['text'].lower().encode('utf-8'))
        labels_val.append(data['reviews'][histLen-1]['stars'])
        ids_val.append(str(data['reviews'][histLen-1]['review_id']))
    ids_val = np.array(ids_val)
    labels_val = np.array(labels_val)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents_train + contents_val)
    textSequences_train = tk.texts_to_sequences(contents_train)
    textVector_train = sequence.pad_sequences(textSequences_train, maxlen=textLength, truncating='post', padding='post')
    textSequences_val = tk.texts_to_sequences(contents_val)
    textVector_val = sequence.pad_sequences(textSequences_val, maxlen=textLength, truncating='post', padding='post')

    if embedding == 'glove':
        embeddings_index = {}
        embFile = open('../tweetEmbeddingData/glove.6B.200d.txt', 'r')
        for line in embFile:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        embFile.close()
        print('Found %s word vectors.' % len(embeddings_index))
        word_index = tk.word_index
        embMatrix = np.zeros((len(word_index) + 1, 200))
        for word, i in word_index.items():
            embVector = embeddings_index.get(word)
            if embVector is not None:
                embMatrix[i] = embVector
    elif embedding == 'word2vec':
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index) + 1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embMatrix[i] = embModel[word]
    else:
        embMatrix = None
        word_index = None

    return ids_train, ids_val, labels_train, labels_val, contents_train, contents_val, textVector_train, textVector_val, embMatrix, word_index


def loadData(modelName, char, embedding, dev=False):
    print('Loading...')
    contents_train = []
    contents_val = []
    labels_train = []
    labels_val = []
    places_train = []
    places_val = []
    ids_train = []
    ids_val = []
    trainFile = open('data/consolidateData_' + modelName + '_train.json', 'r')
    if dev:
        valFile = open('data/consolidateData_' + modelName + '_test.json', 'r')
    else:
        valFile = open('data/consolidateData_' + modelName + '_dev.json', 'r')

    for line in trainFile:
        data = json.loads(line.strip())
        contents_train.append(data['content'].lower().encode('utf-8'))
        labels_train.append(data['label'])
        places_train.append(data['place'])
        ids_train.append(str(data['id']))
    places_train = np.array(places_train)
    ids_train = np.array(ids_train)
    labels_train = np.array(labels_train)

    for line in valFile:
        data = json.loads(line.strip())
        contents_val.append(data['content'].lower().encode('utf-8'))
        labels_val.append(data['label'])
        places_val.append(data['place'])
        ids_val.append(str(data['id']))
    places_val = np.array(places_val)
    ids_val = np.array(ids_val)
    labels_val = np.array(labels_val)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents_train + contents_val)
    tweetSequences_train = tk.texts_to_sequences(contents_train)
    tweetVector_train = sequence.pad_sequences(tweetSequences_train, maxlen=tweetLength, truncating='post', padding='post')
    tweetSequences_val = tk.texts_to_sequences(contents_val)
    tweetVector_val = sequence.pad_sequences(tweetSequences_val, maxlen=tweetLength, truncating='post', padding='post')

    if embedding == 'glove':
        embeddings_index = {}
        embFile = open('../tweetEmbeddingData/glove.twitter.27B.200d.txt', 'r')
        for line in embFile:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        embFile.close()
        print('Found %s word vectors.' % len(embeddings_index))
        word_index = tk.word_index
        embMatrix = np.zeros((len(word_index) + 1, 200))
        for word, i in word_index.items():
            embVector = embeddings_index.get(word)
            if embVector is not None:
                embMatrix[i] = embVector
    elif embedding == 'word2vec':
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index) + 1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embMatrix[i] = embModel[word]
    else:
        embMatrix = None
        word_index = None

    return ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, tweetVector_train, tweetVector_val, embMatrix, word_index


def manageData(trainDataLists, testDataLists):
    print ('Manage train and test data split...')
    output = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    totalDataList = []
    for index, trainData in enumerate(trainDataLists):
        testData = testDataLists[index]
        totalData = np.concatenate((trainData, testData))
        totalDataList.append(totalData)
    for fold, (train_index, test_index) in enumerate(skf.split(totalDataList[4], totalDataList[1])):
        trainDataList = []
        testDataList = []
        for data in totalDataList:
            trainDataList.append(data[train_index])
            testDataList.append(data[test_index])
        output[fold] = (trainDataList, testDataList)

    return output


def processRNN(modelName, RNN='LSTM', balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    if RNN == 'LSTM':
        resultName = 'result/LSTM_' + modelName + '_' + balancedWeight
    elif RNN == 'GRU':
        resultName = 'result/GRU_' + modelName + '_' + balancedWeight

    if 'yelp' in modelName:
        ids_train, ids_val, labels_train, labels_val, contents_train, contents_val, textVector_train, textVector_val, embMatrix, word_index = loadYelpData(modelName, char, embedding)
        textLength = yelpLength
    else:
        ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, textVector_train, textVector_val, embMatrix, word_index = loadData(modelName, char, embedding, dev=dev)
        textLength = tweetLength

    labelNum = len(np.unique(np.concatenate([labels_train, labels_val])))

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([labels_train, labels_val]))
    labels_train = encoder.transform(labels_train)
    labels_val = encoder.transform(labels_val)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    #trainDataList = [ids_train, labels_train, places_train, contents_train, tweetVector_train]
    #testDataList = [ids_val, labels_val, places_val, contents_val, tweetVector_val]

    #expData = manageData(trainDataList, testDataList)

    # training
    if dev:
        verbose = 2
    else:
        verbose = 0
    print('training...')
    eval = evaluation.evalMetrics(labelNum)

    inputs = Input(batch_shape=(batch_size, textLength, ), name='tweet_input')
    if embedding in ['word2vec', 'glove']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(inputs)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(inputs)
    if RNN == 'LSTM':
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn')(embedding_tweet)
    elif RNN == 'GRU':
        tweet_lstm = GRU(300, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn')(embedding_tweet)
    tweet_output = Dense(labelNum, activation='softmax', name='output')(tweet_lstm)
    model = Model(inputs=inputs, outputs=tweet_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model.summary()

    if len(labels_train) % batch_size != 0:
        textVector_train = textVector_train[:-(len(textVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        textVector_val = textVector_val[:-(len(textVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]
        if 'yelp' not in modelName:
            places_val = places_val[:-(len(places_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(textVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(textVector_val, labelVector_val), sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(textVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(textVector_val, labelVector_val), class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(textVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(textVector_val, labelVector_val), verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss)+'\t'+str(accuracyHist[index])+'\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(textVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    predictions = model.predict(textVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        if 'yelp' in modelName:
            sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + str(trueLabel_val[index]) + '\t' + str(predLabel) + '\n')
        else:
            sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

    if not dev:
        score, scoreSTD = eval.getScore()
        precision, preSTD = eval.getPrecision()
        recall, recSTD = eval.getRecall()
        f1, f1STD = eval.getF1()
        conMatrix = eval.getConMatrix()
        resultFile = open(resultName + '.result', 'a')
        confusionFile = open(resultName + '.confMatrix', 'a')
        for row in conMatrix:
            lineOut = ''
            for line in row:
                lineOut += str(line) + '\t'
            confusionFile.write(lineOut.strip() + '\n')
        confusionFile.write('\n')
        resultFile.write(score + '\n')
        resultFile.write(recall + '\n')
        resultFile.write(precision + '\n')
        resultFile.write(f1 + '\n\n')
        confusionFile.close()
        resultFile.close()
        print(score)
        print(recall)
        print(precision)
        print(f1)


def processBiRNN(modelName, RNN='LSTM', balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    if RNN == 'LSTM':
        resultName = 'result/BiLSTM_' + modelName + '_' + balancedWeight
    elif RNN == 'GRU':
        resultName = 'result/BiGRU_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, tweetVector_train, tweetVector_val, embMatrix, word_index = loadData(
        modelName, char, embedding, dev=dev)

    labelNum = len(np.unique(np.concatenate([labels_train, labels_val])))

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([labels_train, labels_val]))
    labels_train = encoder.transform(labels_train)
    labels_val = encoder.transform(labels_val)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    if dev:
        verbose = 2
    else:
        verbose = 0
    print('training...')
    eval = evaluation.evalMetrics(labelNum)

    inputs = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['word2vec', 'glove']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(inputs)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(inputs)
    if RNN == 'LSTM':
        tweet_rnn = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn'))(embedding_tweet)
    elif RNN == 'GRU':
        tweet_rnn = Bidirectional(GRU(300, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn'))(embedding_tweet)
    tweet_output = Dense(labelNum, activation='softmax', name='output')(tweet_rnn)
    model = Model(inputs=inputs, outputs=tweet_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(tweetVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(tweetVector_train, labelVector_train, validation_data=(tweetVector_val, labelVector_val), epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(tweetVector_train, labelVector_train, validation_data=(tweetVector_val, labelVector_val), epochs=epochs, batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(tweetVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict(tweetVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

    if not dev:
        score, scoreSTD = eval.getScore()
        precision, preSTD = eval.getPrecision()
        recall, recSTD = eval.getRecall()
        f1, f1STD = eval.getF1()
        conMatrix = eval.getConMatrix()
        resultFile = open(resultName + '.result', 'a')
        confusionFile = open(resultName + '.confMatrix', 'a')
        for row in conMatrix:
            lineOut = ''
            for line in row:
                lineOut += str(line) + '\t'
            confusionFile.write(lineOut.strip() + '\n')
        confusionFile.write('\n')
        resultFile.write(score + '\t' + scoreSTD + '\n')
        resultFile.write(recall + '\t' + recSTD + '\n')
        resultFile.write(precision + '\t' + preSTD + '\n')
        resultFile.write(f1 + '\t' + f1STD + '\n\n')
        confusionFile.close()
        resultFile.close()
        print(score + ' ' + scoreSTD)
        print(recall + ' ' + recSTD)
        print(precision + ' ' + preSTD)
        print(f1 + ' ' + f1STD)


def processAttRNN(modelName, RNN='LSTM', balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    if RNN == 'LSTM':
        resultName = 'result/AttLSTM_' + modelName + '_' + balancedWeight
    elif RNN == 'GRU':
        resultName = 'result/AttGRU_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, tweetVector_train, tweetVector_val, embMatrix, word_index = loadData(modelName, char, embedding, dev=dev)

    labelNum = len(np.unique(np.concatenate([labels_train, labels_val])))

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([labels_train, labels_val]))
    labels_train = encoder.transform(labels_train)
    labels_val = encoder.transform(labels_val)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    #trainDataList = [ids_train, labels_train, places_train, contents_train, tweetVector_train]
    #testDataList = [ids_val, labels_val, places_val, contents_val, tweetVector_val]
    #expData = manageData(trainDataList, testDataList)

    # training
    if dev:
        verbose = 2
    else:
        verbose = 0
    print('training...')
    eval = evaluation.evalMetrics(labelNum)

    inputs = Input(batch_shape=(batch_size, tweetLength, ), name='tweet_input')
    if embedding in ['word2vec', 'glove']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(inputs)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(inputs)
    if RNN == 'LSTM':
        tweet_rnn = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn', return_sequences=True)(embedding_tweet)
    elif RNN == 'GRU':
        tweet_rnn = GRU(300, dropout=0.2, recurrent_dropout=0.2, name='tweet_rnn', return_sequences=True)(embedding_tweet)
    self_attention = SeqSelfAttention(attention_activation='sigmoid', name='self_attention')(tweet_rnn)
    #flatten_result = Flatten()(self_attention)
    last_timestep = Lambda(lambda x: x[:, -1, :])(self_attention)
    tweet_output = Dense(labelNum, activation='softmax', name='output')(last_timestep)
    model = Model(inputs=inputs, outputs=tweet_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(tweetVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweetVector_val, labelVector_val), sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(tweetVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweetVector_val, labelVector_val), class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(tweetVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweetVector_val, labelVector_val), verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss)+'\t'+str(accuracyHist[index])+'\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(tweetVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
    predictions = model.predict(tweetVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

    if not dev:
        score, scoreSTD = eval.getScore()
        precision, preSTD = eval.getPrecision()
        recall, recSTD = eval.getRecall()
        f1, f1STD = eval.getF1()
        conMatrix = eval.getConMatrix()
        resultFile = open(resultName + '.result', 'a')
        confusionFile = open(resultName + '.confMatrix', 'a')
        for row in conMatrix:
            lineOut = ''
            for line in row:
                lineOut += str(line) + '\t'
            confusionFile.write(lineOut.strip() + '\n')
        confusionFile.write('\n')
        resultFile.write(score + '\t' + scoreSTD + '\n')
        resultFile.write(recall + '\t' + recSTD + '\n')
        resultFile.write(precision + '\t' + preSTD + '\n')
        resultFile.write(f1 + '\t' + f1STD + '\n\n')
        confusionFile.close()
        resultFile.close()
        print(score + ' ' + scoreSTD)
        print(recall + ' ' + recSTD)
        print(precision + ' ' + preSTD)
        print(f1 + ' ' + f1STD)


def processCNNLSTM(modelName, RNN='LSTM', balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    if RNN == 'LSTM':
        resultName = 'result/CNNLSTM_' + modelName + '_' + balancedWeight
    elif RNN == 'GRU':
        resultName = 'result/CNNGRU_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, tweetVector_train, tweetVector_val, embMatrix, word_index = loadData(
        modelName, char, embedding, dev=dev)

    labelNum = len(np.unique(np.concatenate([labels_train, labels_val])))

    encoder = LabelEncoder()
    encoder.fit(np.concatenate([labels_train, labels_val]))
    labels_train = encoder.transform(labels_train)
    labels_val = encoder.transform(labels_val)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    if dev:
        verbose = 2
    else:
        verbose = 0
    print('training...')
    eval = evaluation.evalMetrics(labelNum)

    input = Input(batch_shape=(batch_size, tweetLength, ))
    if embedding in ['word2vec', 'glove']:
        embedding_tweet = Embedding(len(word_index)+1, 200, weights=[embMatrix], trainable=True)(input)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input)
    tweet_cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedding_tweet)
    tweet_pool = MaxPooling1D(pool_size=2)(tweet_cnn)
    if RNN == 'LSTM':
        tweet_rnn = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='LSTM')(tweet_pool)
    elif RNN == 'GRU':
        tweet_rnn = GRU(300, dropout=0.2, recurrent_dropout=0.2, name='GRU')(tweet_pool)
    output = Dense(labelNum, activation='softmax')(tweet_rnn)
    model = Model(inputs=input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(tweetVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(tweetVector_train, labelVector_train, validation_data=(tweetVector_val, labelVector_val), epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(tweetVector_train, labelVector_train,
                  validation_data=(tweetVector_val, labelVector_val), epochs=epochs, batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(tweetVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict(tweetVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

    if not dev:
        score, scoreSTD = eval.getScore()
        precision, preSTD = eval.getPrecision()
        recall, recSTD = eval.getRecall()
        f1, f1STD = eval.getF1()
        conMatrix = eval.getConMatrix()
        resultFile = open(resultName + '.result', 'a')
        confusionFile = open(resultName + '.confMatrix', 'a')
        for row in conMatrix:
            lineOut = ''
            for line in row:
                lineOut += str(line) + '\t'
            confusionFile.write(lineOut.strip() + '\n')
        confusionFile.write('\n')
        resultFile.write(score + '\n')
        resultFile.write(recall + '\n')
        resultFile.write(precision + '\n')
        resultFile.write(f1 + '\n\n')
        confusionFile.close()
        resultFile.close()
        print(score)
        print(recall)
        print(precision)
        print(f1)



if __name__ == "__main__":
    modelName = 'long1.5'
    #modelName = 'yelpUserReview'
    embModel = 'glove'

    #processRNN(modelName, RNN='GRU', balancedWeight='class', embedding=embModel, char=False, epochs=5, dev=False)
    #processBiRNN(modelName, RNN='GRU', balancedWeight='class', embedding=embModel, char=False, epochs=3, dev=False)
    #processAttRNN(modelName, RNN='GRU', balancedWeight='class', embedding=embModel, char=False, epochs=4, dev=False)
    processCNNLSTM(modelName, RNN='GRU', balancedWeight='class', embedding=embModel, char=False, epochs=4, dev=False)
