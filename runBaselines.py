import json, re
import numpy as np
from keras.layers import Dense, LSTM, Conv1D, Flatten, Reshape, Bidirectional, Input, MaxPooling1D, dot, add
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from utilities import word2vecReader
from sklearn.metrics import precision_score, recall_score, f1_score
from wordsegment import load

vocabSize = 10000
tweetLength = 25
embeddingVectorLength = 200
charLengthLimit = 20
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


def loadData(modelName, char, embedding):
    print('Loading...')
    contents = []
    labels = []
    places = []
    dataFile = open('data/consolidateData_'+modelName+'.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        contents.append(data['content'].encode('utf-8'))
        labels.append(data['label'])
        places.append(data['place'])

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

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

    places = np.array(places)

    return labels, places, contents, tweetVector, embMatrix, word_index


def processLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, tweetVector, embMatrix, word_index = loadData(modelName, char, embedding)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    if tune:
        verbose = 2
    else:
        verbose = 0
    print('training...')
    resultFile = open('result/LSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]

        inputs = Input(batch_shape=(batch_size, tweetLength, ), name='tweet_input')
        if embedding in ['word2vec', 'glove']:
            embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(inputs)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(inputs)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='tweet_lstm')(embedding_tweet)
        tweet_output = Dense(labelNum, activation='softmax', name='output')(tweet_lstm)
        model = Model(inputs=inputs, outputs=tweet_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweet_test, labelVector_test), sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweet_test, labelVector_test), class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweet_test, labelVector_test), verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/LSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index) + '\t' + str(loss)+'\t'+str(accuracyHist[index])+'\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate(tweet_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        score += scores[1] * 100
        predictions = model.predict(tweet_test, batch_size=batch_size)
        sampleFile = open('result/LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            if index % 100 == 0:
                sampleFile.write(contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\t' + places_test[index] + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        precision += precision_score(labels_test, predLabels, average='macro')
        recall += recall_score(labels_test, predLabels, average='macro')
        f1 += f1_score(labels_test, predLabels, average='macro')
        conMatrix = confusion_matrix(labels_test, predLabels)
        sumConMatrix = np.add(sumConMatrix, conMatrix)
        if tune:
            break

    sumConMatrix = np.divide(sumConMatrix, 5)
    confusionFile = open('result/LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
    for row in sumConMatrix:
        lineOut = ''
        for line in row:
            lineOut += str(line) + '\t'
        confusionFile.write(lineOut.strip() + '\n')
    confusionFile.write('\n')
    confusionFile.close()
    resultFile.write(str(score / 5) + '\n')
    resultFile.write(str(recall * 100 / 5) + '\n')
    resultFile.write(str(precision * 100 / 5) + '\n')
    resultFile.write(str(f1 * 100 / 5) + '\n\n')
    print(score / 5)
    print(recall * 100 / 5)
    print(precision*100 / 5)
    print(f1*100 / 5)
    resultFile.close()


def processBiLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, tweetVector, embMatrix, word_index = loadData(modelName, char, embedding)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/BiLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()
    places = np.array(places)

    # training
    print('training...')
    resultFile = open('result/BiLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        inputs = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['word2vec', 'glove']:
            embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(inputs)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(inputs)
        tweet_lstm = Bidirectional(LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='tweet_lstm'))(embedding_tweet)
        tweet_output = Dense(labelNum, activation='softmax', name='output')(tweet_lstm)
        model = Model(inputs=inputs, outputs=tweet_output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(tweet_train, labelVector_train, validation_data=(tweet_test, labelVector_test), epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=2)
        else:
            model.fit(tweet_train, labelVector_train, validation_data=(tweet_test, labelVector_test), epochs=epochs, batch_size=batch_size, verbose=2)

        scores = model.evaluate(tweet_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(tweet_test, batch_size=batch_size)
        sampleFile = open('result/BiLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            if index % 100 == 0:
                sampleFile.write(contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        precision += precision_score(labels_test, predLabels, average='macro')
        recall += recall_score(labels_test, predLabels, average='macro')
        f1 += f1_score(labels_test, predLabels, average='macro')
        conMatrix = confusion_matrix(labels_test, predLabels)
        sumConMatrix = np.add(sumConMatrix, conMatrix)
        if tune:
            break

    sumConMatrix = np.divide(sumConMatrix, 5)
    confusionFile = open('result/BiLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
    for row in sumConMatrix:
        lineOut = ''
        for line in row:
            lineOut += str(line) + '\t'
        confusionFile.write(lineOut.strip() + '\n')
    confusionFile.write('\n')
    confusionFile.close()
    resultFile.write(str(score / 5) + '\n')
    resultFile.write(str(recall * 100 / 5) + '\n')
    resultFile.write(str(precision * 100 / 5) + '\n')
    resultFile.write(str(f1 * 100 / 5) + '\n\n')
    print(score / 5)
    print(recall*100 / 5)
    print(precision*100 / 5)
    print(f1*100 / 5)
    resultFile.close()


def processAttLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, tweetVector, embMatrix, word_index = loadData(modelName, char, embedding)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/AttLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()
    places = np.array(places)

    # training
    print('training...')
    score = 0.0
    f1 = 0.0
    precision = 0.0
    recall = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    resultFile = open('result/AttLSTM_'+modelName+'_'+balancedWeight+'.result', 'a')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        tweet_input1 = Input(batch_shape=(batch_size, tweetLength,))
        tweet_input2 = Input(batch_shape=(batch_size, tweetLength,))
        if embedding in ['word2vec', 'glove']:
            tweet_embedding1 = Embedding(len(word_index)+1, 200, weights=[embMatrix], trainable=True, input_length=tweetLength)(tweet_input1)
            tweet_embedding2 = Embedding(len(word_index)+1, 200, weights=[embMatrix], trainable=True, input_length=tweetLength)(tweet_input2)
        else:
            tweet_embedding1 = Embedding(vocabSize, embeddingVectorLength, name='emb1')(tweet_input1)
            tweet_embedding2 = Embedding(vocabSize, embeddingVectorLength, name='emb2')(tweet_input2)

        lstm1 = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='LSTM1')(tweet_embedding1)
        lstm2 = LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='LSTM2')(tweet_embedding2)

        tweet_dot = dot([lstm1, lstm2], axes=[1, 1])
        tweet_flatten = Flatten()(tweet_dot)
        tweet_dense = Dense(tweetLength * 100, activation='relu')(tweet_flatten)
        tweet_reshape = Reshape((tweetLength, 100))(tweet_dense)

        tweet_sum = add([lstm1, tweet_reshape])
        tweet_flatten = Flatten()(tweet_sum)
        output = Dense(labelNum, activation='softmax')(tweet_flatten)
        model = Model(inputs=[tweet_input1, tweet_input2], outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labelVector_train)
            model.fit([tweet_train, tweet_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, tweet_train], labelVector_train, validation_data=([tweet_test, tweet_test], labelVector_test), epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=2)
        else:
            model.fit([tweet_train, tweet_train], labelVector_train, validation_data=([tweet_test, tweet_test], labelVector_test), epochs=epochs, batch_size=batch_size, verbose=2)

        scores = model.evaluate([tweet_test, tweet_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        score += scores[1] * 100
        predictions = model.predict([tweet_test, tweet_test], batch_size=batch_size)
        sampleFile = open('result/AttLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            if index % 100 == 0:
                sampleFile.write(contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        precision += precision_score(labels_test, predLabels, average='macro')
        recall += recall_score(labels_test, predLabels, average='macro')
        f1 += f1_score(labels_test, predLabels, average='macro')
        conMatrix = confusion_matrix(labels_test, predLabels)
        sumConMatrix = np.add(sumConMatrix, conMatrix)
        if tune:
            break

    sumConMatrix = np.divide(sumConMatrix, 5)
    confusionFile = open('result/AttLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
    for row in sumConMatrix:
        lineOut = ''
        for line in row:
            lineOut += str(line) + '\t'
        confusionFile.write(lineOut.strip() + '\n')
    confusionFile.write('\n')
    confusionFile.close()
    resultFile.write(str(score / 5) + '\n')
    resultFile.write(str(recall * 100 / 5) + '\n')
    resultFile.write(str(precision * 100 / 5) + '\n')
    resultFile.write(str(f1 * 100 / 5) + '\n\n')
    print(score / 5)
    print(recall*100 / 5)
    print(precision*100 / 5)
    print(f1*100 / 5)
    resultFile.close()


def processCNNLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, tweetVector, embMatrix, word_index = loadData(modelName, char, embedding)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()
    places = np.array(places)

    # training
    print('training...')
    score = 0.0
    f1 = 0.0
    precision = 0.0
    recall = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    resultFile = open('result/CNNLSTM_'+modelName+'_'+balancedWeight+'.result', 'a')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input = Input(batch_shape=(batch_size, tweetLength, ))
        if embedding in ['word2vec', 'glove']:
            embedding_tweet = Embedding(len(word_index)+1, 200, weights=[embMatrix], trainable=True)(input)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input)
        tweet_cnn = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(embedding_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='LSTM')(tweet_cnn)
        output = Dense(labelNum, activation='softmax')(tweet_lstm)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(tweet_train, labelVector_train, validation_data=(tweet_test, labelVector_test), epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=2)
        else:
            model.fit(tweet_train, labelVector_train,
                      validation_data=(tweet_test, labelVector_test), epochs=epochs, batch_size=batch_size, verbose=2)

        scores = model.evaluate(tweet_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        score += scores[1] * 100
        predictions = model.predict(tweet_test, batch_size=batch_size)
        sampleFile = open('result/CNNLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            if index % 100 == 0:
                sampleFile.write(contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        precision += precision_score(labels_test, predLabels, average='macro')
        recall += recall_score(labels_test, predLabels, average='macro')
        f1 += f1_score(labels_test, predLabels, average='macro')
        conMatrix = confusion_matrix(labels_test, predLabels)
        sumConMatrix = np.add(sumConMatrix, conMatrix)
        if tune:
            break

    sumConMatrix = np.divide(sumConMatrix, 5)
    confusionFile = open('result/CNNLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
    for row in sumConMatrix:
        lineOut = ''
        for line in row:
            lineOut += str(line) + '\t'
        confusionFile.write(lineOut.strip() + '\n')
    confusionFile.write('\n')
    confusionFile.close()
    resultFile.write(str(score / 5) + '\n')
    resultFile.write(str(recall * 100 / 5) + '\n')
    resultFile.write(str(precision * 100 / 5) + '\n')
    resultFile.write(str(f1 * 100 / 5) + '\n\n')
    print(score / 5)
    print(recall*100 / 5)
    print(precision*100 / 5)
    print(f1*100 / 5)
    resultFile.close()

'''
def processLSTMCNN(modelName, balancedWeight='None', embedding='None', char=False, epochs=4):
    activityList = []
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for line in activityListFile:
        if not line.startswith('#'):
            activityList.append(line.strip())
    activityListFile.close()
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()
    labelNum = len(np.unique(activityList))
    if 'NONE' in activityList:
        labelNum -= 1

    contents = []
    labels = []
    labelCount = {}
    timeList = []
    labelTweetCount = {}
    placeTweetCount = {}
    labelLabel = {}
    labelContent = {}
    labelTime = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            if activity not in labelTweetCount:
                labelTweetCount[activity] = 0.0
            tweetFile = open('data/POIplace/' + place + '.json', 'r')
            tweetCount = 0
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8')
                    contents.append(content)
                    dateTemp = data['created_at'].split()
                    time = [dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])]
                    timeList.append(time)
                    labels.append(activity)
                    tweetCount += 1
                    if activity not in labelCount:
                        labelCount[activity] = 0.0
                    labelCount[activity] += 1.0
                    if activity not in labelLabel:
                        labelLabel[activity] = []
                    labelLabel[activity].append(activity)
                    if activity not in labelContent:
                        labelContent[activity] = []
                    labelContent[activity].append(content)
                    if activity not in labelTime:
                        labelTime[activity] = []
                    labelTime[activity].append(time)
            tweetFile.close()
            labelTweetCount[activity] += tweetCount
            placeTweetCount[place] = tweetCount

    timeVector = np.array(timeList)
    encoder = LabelEncoder()
    labels = np.array(labels)
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    labelFile = open('result/LSTMCNN_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, padding='post', truncating='post')

    if embedding == 'glove':
        print ('Loading glove embeddings...')
        embeddings_index = {}
        embFile = open('../tweetEmbeddingData/glove.twitter.27B.100d.txt', 'r')
        for line in embFile:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        embFile.close()
        print('Found %s word vectors.' % len(embeddings_index))
        word_index = tk.word_index
        embMatrix = np.zeros((len(word_index) + 1, 100))
        for word, i in word_index.items():
            if word in embeddings_index:
                embVector = embeddings_index[word]
                embMatrix[i] = embVector
    elif embedding == 'word2vec':
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index)+1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embVector = embModel[word]
                embMatrix[i] = embVector

    # training
    print('training...')
    score = 0.0
    f1 = 0.0
    precision = 0.0
    recall = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    resultFile = open('result/LSTMCNN_'+modelName+'_'+balancedWeight+'.result', 'a')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(timeVector, labels)):
        input = Input(shape=(tweetLength,))
        if embedding in ['word2vec', 'glove']:
            embedding_tweet = Embedding(len(word_index) + 1, 400, weights=[embMatrix], trainable=False)(input)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input)
        print(embedding_tweet._keras_shape)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='LSTM', return_sequences=True)(embedding_tweet)
        print(tweet_lstm._keras_shape)
        tweet_cnn = TimeDistributed(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))(tweet_lstm)
        tweet_pooling = TimeDistributed(MaxPooling1D(pool_size=4))(tweet_cnn)
        print tweet_pooling._keras_shape
        output = Dense(labelNum, activation='softmax')(tweet_pooling)
        model = Model(inputs=input, outputs=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # print(model.summary())

        tweet_train = tweetVector[train_index]
        time_train = timeVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=10, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=10, class_weight=classWeight, verbose=0)
        elif balancedWeight == 'class_label':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(countSum / labelCount[label])
            model.fit([tweet_train, time_train], labelVector_train, epochs=epochs, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label_log':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(-math.log(labelCount[label] / countSum))
            model.fit([tweet_train, time_train], labelVector_train, epochs=epochs, batch_size=10, class_weight=classWeight)
        else:
            model.fit(tweet_train, labelVector_train,
                      validation_data=(tweet_test, labelVector_test), epochs=epochs, batch_size=10, verbose=0)

        scores = model.evaluate(tweet_test, labelVector_test, batch_size=1, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        score += scores[1] * 100
        predictions = model.predict(tweet_test)
        sampleFile = open('result/LSTMCNN_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            if index % 100 == 0:
                sampleFile.write(contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        precision += precision_score(labels_test, predLabels, average='macro')
        recall += recall_score(labels_test, predLabels, average='macro')
        f1 += f1_score(labels_test, predLabels, average='macro')

        conMatrix = confusion_matrix(labels_test, predLabels)
        sumConMatrix = np.add(sumConMatrix, conMatrix)

    sumConMatrix = np.divide(sumConMatrix, 5)
    confusionFile = open('result/LSTMCNN_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
    for row in sumConMatrix:
        lineOut = ''
        for line in row:
            lineOut += str(line) + '\t'
        confusionFile.write(lineOut.strip() + '\n')
    confusionFile.write('\n')
    confusionFile.close()
    resultFile.write(str(score / 5) + '\n')
    resultFile.write(str(recall * 100 / 5) + '\n')
    resultFile.write(str(precision * 100 / 5) + '\n')
    resultFile.write(str(recall * 100 / 5) + '\n\n')
    print(score / 5)
    print(recall * 100 / 5)
    print(precision*100 / 5)
    print(f1*100 / 5)
    resultFile.close()
'''

if __name__ == "__main__":
    processLSTM('long1.5', 'none', 'glove', char=False, epochs=9, tune=False)
    processLSTM('long1.5', 'class', 'glove', char=False, epochs=6, tune=False)

    #processBiLSTM('long1.5', 'none', 'glove', char=False, epochs=5, tune=False)
    ## re-run the following one from tuning
    #processBiLSTM('long1.5', 'class', 'glove', char=False, epochs=5, tune=False)

    #processCNNLSTM('long1.5', 'none', 'glove', char=False, epochs=10, tune=False)
    #processCNNLSTM('long1.5', 'class', 'glove', char=False, epochs=10, tune=False)

    #processAttLSTM('long1.5', 'none', 'glove', char=False, epochs=4, tune=False)
    #processAttLSTM('long1.5', 'class', 'glove', char=False, epochs=9, tune=False)
