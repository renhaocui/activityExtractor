import json, re
import numpy as np
from keras.layers import Dense, LSTM, Input, concatenate, Lambda, multiply, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils, to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from utilities import word2vecReader
from keras import backend as K
from wordsegment import load, segment
from utilities import tokenizer
load()

vocabSize = 10000
tweetLength = 25
posEmbLength = 25
embeddingVectorLength = 200
embeddingPOSVectorLength = 20
charLengthLimit = 20
batch_size = 100

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}
POSMapper = {'N': 'N', 'O': 'N', '^': 'N', 'S': 'N', 'Z': 'N', 'L': 'N', 'M': 'N',
             'V': 'V', 'A': 'A', 'R': 'R', '@': '@', '#': '#', '~': '~', 'E': 'E', ',': ',', 'U': 'U',
             '!': '0', 'D': '0', 'P': '0', '&': '0', 'T': '0', 'X': '0', 'Y': '0', '$': '0', 'G': '0'}

def sequential_output_shape(input_shape):
    return (input_shape[0][0], 2, input_shape[0][1])


def sequential_concat(x):
    return K.stack([x[0], x[1]])


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


def cleanContent(input, hashtag=True, breakEmoji=True):
    #print input
    input = input.replace('\n', ' ').replace('\r', ' ').replace('#', ' #')
    input = removeLinks(input)
    output = ''
    for word in tokenizer.simpleTokenize(input):
        if breakEmoji:
            emojis1 = re.findall(r'\\u....', word.encode('unicode-escape'))
            emojis2 = re.findall(r'\\U........', word.encode('unicode-escape'))
            emojis = emojis1 + emojis2
        if (not hashtag) and word.startswith('#'):
            segTemp = segment(word[1:])
            for seg in segTemp:
                output += seg + ' '
        elif len(emojis) > 0:
            for emoji in emojis:
                output += emoji + ' '
        else:
            output += word + ' '

    return output.strip().encode('utf-8')


def removeLinks(input):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, '')
    return input


def extractVerb(inputs):
    output = []
    for content in inputs:
        if content[1] == 'V':
            output.append(content[0])
    return output


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


def loadData(modelName, char, embedding, pos=False):
    print('Loading...')
    contents = []
    labels = []
    places = []
    days = []
    hours = []
    poss = []
    dataFile = open('data/consolidateData_'+modelName+'.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        contents.append(data['content'].encode('utf-8'))
        labels.append(data['label'])
        places.append(data['place'])
        days.append(data['day'])
        hours.append(data['hour'])
        poss.append(data['pos'].encode('utf-8'))

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

    if not pos:
        return labels, places, contents, days, hours, tweetVector, embMatrix, word_index
    else:
        posVocabSize = 25
        tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
        tkPOS.fit_on_texts(poss)
        posSequences = tkPOS.texts_to_sequences(poss)
        posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

        return labels, places, contents, days, hours, poss, tweetVector, posVector, posVocabSize, embMatrix, word_index


def processTLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, dayList, hourList, tweetVector, embMatrix, word_index = loadData(modelName, char, embedding, pos=False)

    labelNum = len(np.unique(labels))
    dayVector = to_categorical(dayList, num_classes=7)
    hourVector = to_categorical(hourList, num_classes=4)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-TLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    print('training...')
    if tune:
        verbose = 2
    else:
        verbose = 0
    resultFile = open('result/C-TLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_tweet)

        input_day = Input(batch_shape=(batch_size, 7,))
        day_dense = Dense(20, activation='relu', name='day')(input_day)

        input_hour = Input(batch_shape=(batch_size, 4,))
        hour_dense = Dense(20, activation='relu', name='hour')(input_hour)

        comb = concatenate([tweet_lstm, hour_dense, day_dense])
        output = Dense(labelNum, activation='softmax', name='output')(comb)
        model = Model(inputs=[input_tweet, input_hour, input_day], outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        day_train = dayVector[train_index]
        hour_train = hourVector[train_index]
        labels_train = labels[train_index]

        tweet_test = tweetVector[test_index]
        day_test = dayVector[test_index]
        hour_test = hourVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = np.array(places)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            hour_train = hour_train[:-(len(hour_train) % batch_size)]
            day_train = day_train[:-(len(day_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            hour_test = hour_test[:-(len(hour_test) % batch_size)]
            day_test = day_test[:-(len(day_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit([tweet_train, hour_train, day_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit([tweet_train, hour_train, day_train], labelVector_train, validation_data=([tweet_test, hour_test, day_test], labelVector_test), epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit([tweet_train, hour_train, day_train], labelVector_train, validation_data=([tweet_test, hour_test, day_test], labelVector_test), epochs=epochs, batch_size=batch_size, verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/C-TLSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate([tweet_test, hour_test, day_test], labelVector_test, batch_size=batch_size, verbose=2)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, hour_test, day_test], batch_size=batch_size)
        sampleFile = open('result/C-TLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-TLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processPOSLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, dayList, hourList, posList, tweetVector, posVector, posVocabSize, embMatrix, word_index = loadData(modelName, char, embedding, pos=True)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-POSLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    print('training...')
    if tune:
        verbose = 2
    else:
        verbose = 0
    resultFile = open('result/C-POSLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_tweet)

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)
        pos_lstm = LSTM(20, dropout=0.2, recurrent_dropout=0.2)(embedding_pos)

        comb = concatenate([tweet_lstm, pos_lstm])
        output = Dense(labelNum, activation='softmax', name='output')(comb)
        model = Model(inputs=[input_tweet, input_pos], outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        posVector_train = posVector[train_index]
        posVector_test = posVector[test_index]
        places_test = np.array(places)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            posVector_test = posVector_test[:-(len(posVector_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test], labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test], labelVector_test), batch_size=batch_size, verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/C-POSLSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate([tweet_test, posVector_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, posVector_test], batch_size=batch_size)
        sampleFile = open('result/C-POSLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-POSLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processPOSTLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, tune=False):
    labels, places, contents, dayList, hourList, posList, tweetVector, posVector, posVocabSize, embMatrix, word_index = loadData(modelName, char, embedding, pos=True)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-POSTLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()
    dayVector = to_categorical(dayList, num_classes=7)
    hourVector = to_categorical(hourList, num_classes=4)

    # training
    print('training...')
    if tune:
        verbose = 2
    else:
        verbose = 0
    resultFile = open('result/C-POSTLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_tweet)

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)
        pos_lstm = LSTM(20, dropout=0.2, recurrent_dropout=0.2)(embedding_pos)

        input_day = Input(batch_shape=(batch_size, 7,))
        day_dense = Dense(20, activation='relu', name='day')(input_day)

        input_hour = Input(batch_shape=(batch_size, 4,))
        hour_dense = Dense(20, activation='relu', name='hour')(input_hour)

        comb = concatenate([tweet_lstm, day_dense, hour_dense, pos_lstm])
        output = Dense(labelNum, activation='softmax', name='output')(comb)
        model = Model(inputs=[input_tweet, input_day, input_hour, input_pos], outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        day_train = dayVector[train_index]
        hour_train = hourVector[train_index]

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        day_test = dayVector[test_index]
        hour_test = hourVector[test_index]
        contents_test = np.array(contents)[test_index]
        posVector_train = posVector[train_index]
        posVector_test = posVector[test_index]
        places_test = np.array(places)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            hour_train = hour_train[:-(len(hour_train) % batch_size)]
            day_train = day_train[:-(len(day_train) % batch_size)]
            posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            hour_test = hour_test[:-(len(hour_test) % batch_size)]
            day_test = day_test[:-(len(day_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            posVector_test = posVector_test[:-(len(posVector_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit([tweet_train, day_train, hour_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit([tweet_train, day_train, hour_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, day_test, hour_test, posVector_test], labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit([tweet_train, day_train, hour_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, day_test, hour_test, posVector_test], labelVector_test), batch_size=batch_size, verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/C-POSTLSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate([tweet_test, day_test, hour_test, posVector_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, day_test, hour_test, posVector_test], batch_size=batch_size)
        sampleFile = open('result/C-POSTLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-POSTLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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

'''
#multiply after LSTM
def processMLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', epochs=4):
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

    if posMode == 'all':
        posVocabSize = 25
    else:
        posVocabSize = 11

    contents = []
    labels = []
    posList = []
    labelCount = {}
    idTagMapper = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            POSFile = open('data/POS/' + place + '.pos', 'r')
            for line in POSFile:
                data = json.loads(line.strip())
                idTagMapper[data['id']] = data['tag']
            POSFile.close()

            tweetFile = open('data/POIplace/' + place + '.json', 'r')
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    id = data['id']
                    # content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                    content, pos = extractPOS(idTagMapper[id], posMode)
                    contents.append(content)
                    # pos = extractPOS(idTagMapper[id], posMode)[1]
                    posList.append(pos)
                    labels.append(activity)
                    if activity not in labelCount:
                        labelCount[activity] = 0.0
                    labelCount[activity] += 1.0
            tweetFile.close()

    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/MLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    tkPOS.fit_on_texts(posList)
    posSequences = tkPOS.texts_to_sequences(posList)
    posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

    if embedding == 'glove':
        print ('Loading glove embeddings...')
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
            if word in embeddings_index:
                embVector = embeddings_index[word]
                embMatrix[i] = embVector
    elif embedding == 'word2vec':
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index) + 1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embMatrix[i] = embModel[word]

    # training
    print('training...')
    resultFile = open('result/MLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            Embedding(len(word_index) + 1, embeddingVectorLength, weights=[embMatrix], trainable=False)(input_tweet)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_tweet)
  
        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)
        pos_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_pos)

        comb = multiply([tweet_lstm, pos_lstm])
        output = Dense(labelNum, activation='softmax', name='output')(comb)
        model = Model(inputs=[input_tweet, input_pos], outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        posVector_train = posVector[train_index]
        posVector_test = posVector[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            posVector_test = posVector_test[:-(len(posVector_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test], labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate([tweet_test, posVector_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, posVector_test], batch_size=batch_size)
        sampleFile = open('result/MLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/MLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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

# multiply after embedding
def processMLSTM2(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', epochs=4):
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

    if posMode == 'all':
        posVocabSize = 25
    else:
        posVocabSize = 11

    contents = []
    labels = []
    posList = []
    labelCount = {}
    idTagMapper = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            POSFile = open('data/POS/' + place + '.pos', 'r')
            for line in POSFile:
                data = json.loads(line.strip())
                idTagMapper[data['id']] = data['tag']
            POSFile.close()

            tweetFile = open('data/POIplace/' + place + '.json', 'r')
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    id = data['id']
                    # content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                    content, pos = extractPOS(idTagMapper[id], posMode)
                    contents.append(content)
                    # pos = extractPOS(idTagMapper[id], posMode)[1]
                    posList.append(pos)
                    labels.append(activity)
                    if activity not in labelCount:
                        labelCount[activity] = 0.0
                    labelCount[activity] += 1.0
            tweetFile.close()

    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/MLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    tkPOS.fit_on_texts(posList)
    posSequences = tkPOS.texts_to_sequences(posList)
    posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

    if embedding == 'glove':
        print ('Loading glove embeddings...')
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
            if word in embeddings_index:
                embVector = embeddings_index[word]
                embMatrix[i] = embVector
    elif embedding == 'word2vec':
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index) + 1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embMatrix[i] = embModel[word]

    # training
    print('training...')
    resultFile = open('result/MLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            Embedding(len(word_index) + 1, embeddingVectorLength, weights=[embMatrix], trainable=False)(input_tweet)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)
        input_comb = multiply([embedding_tweet, embedding_pos])

        lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(input_comb)
        output = Dense(labelNum, activation='softmax', name='output')(lstm)
        model = Model(inputs=[input_tweet, input_pos], outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        posVector_train = posVector[train_index]
        posVector_test = posVector[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            posVector_test = posVector_test[:-(len(posVector_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test], labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate([tweet_test, posVector_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, posVector_test], batch_size=batch_size)
        sampleFile = open('result/MLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/MLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', epochs=4):
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

    if posMode == 'all':
        posVocabSize = 25
    else:
        posVocabSize = 11

    contents = []
    labels = []
    posList = []
    labelCount = {}
    dayList = []
    hourList = []
    idTagMapper = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            POSFile = open('data/POS/' + place + '.pos', 'r')
            for line in POSFile:
                data = json.loads(line.strip())
                idTagMapper[data['id']] = data['tag']
            POSFile.close()

            tweetFile = open('data/POIplace/' + place + '.json', 'r')
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                    contents.append(tweetTextCleaner.tweetCleaner(content))
                    id = data['id']
                    posList.append((extractPOS(idTagMapper[id], posMode)).encode('utf-8'))
                    dateTemp = data['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    dayList.append(day)
                    hourList.append(hour)
                    labels.append(activity)
                    if activity not in labelCount:
                        labelCount[activity] = 0.0
                    labelCount[activity] += 1.0
            tweetFile.close()

    dayVector = to_categorical(dayList, num_classes=7)
    hourVector = to_categorical(hourList, num_classes=4)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/HLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    tkPOS.fit_on_texts(posList)
    posSequences = tkPOS.texts_to_sequences(posList)
    posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

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
        embMatrix = np.zeros((len(word_index) + 1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embMatrix[i] = embModel[word]

    # training
    print('training...')
    resultFile = open('result/HLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dayVector, labels)):
        input_tweet = Input(shape=(tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            Embedding(len(word_index) + 1, 400, weights=[embMatrix], trainable=False)(input_tweet)
        else:
            embedding_tweet = Embedding(vocabSize, embeddingVectorLength, name='embedding_tweet')(input_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='lstm_tweet')(embedding_tweet)
        
        input_pos = Input(shape=(posEmbLength,), name='tweet_pos')
        embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength, name='embedding_pos')(input_pos)
        pos_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='lstm_pos')(embedding_pos)

        #tuple_lstm = concatenate([pos_lstm, tweet_lstm], axis=0)
        comb = Lambda(sequential_concat, output_shape=sequential_output_shape, name='sequential_merge')([pos_lstm, tweet_lstm])
        print(tweet_lstm._keras_shape)
        print(pos_lstm._keras_shape)
        print(comb._keras_shape)
        comb_lstm = LSTM(100, dropout=0.2, recurrent_dropout=0.2, name='lstm_comb')(comb)
        output = Dense(labelNum, activation='softmax', name='output')(comb_lstm)
        
        model = Model(inputs=[input_tweet, input_pos], outputs=output)
        print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        day_train = dayVector[train_index]
        hour_train = hourVector[train_index]
        labels_train = labels[train_index]

        tweet_test = tweetVector[test_index]
        day_test = dayVector[test_index]
        hour_test = hourVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        posVector_train = posVector[train_index]
        posVector_test = posVector[test_index]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit([tweet_train, hour_train, day_train, posVector_train], labelVector_train, epochs=epochs, batch_size=10,
                      sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, hour_train, day_train, posVector_train], labelVector_train, epochs=epochs, batch_size=10,
                      class_weight=classWeight, verbose=0)
        elif balancedWeight == 'class_label':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(countSum / labelCount[label])
            model.fit([tweet_train, hour_train, day_train], labelVector_train, epochs=epochs, batch_size=10,
                      class_weight=classWeight)
        elif balancedWeight == 'class_label_log':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(-math.log(labelCount[label] / countSum))
            model.fit([tweet_train, hour_train, day_train], labelVector_train, epochs=epochs, batch_size=10,
                      class_weight=classWeight)
        else:
            model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test], labelVector_test), batch_size=10,
                      verbose=2)

        scores = model.evaluate([tweet_test, posVector_test], labelVector_test, batch_size=1, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, posVector_test])
        sampleFile = open('result/HLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/HLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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
    #processTLSTM('long1.5', 'none', 'glove', char=False, epochs=8, tune=False)
    #processTLSTM('long1.5', 'class', 'glove', char=False, epochs=7, tune=False)

    #processPOSLSTM('long1.5', 'none', 'glove', char=False, epochs=8, tune=False)
    #processPOSLSTM('long1.5', 'class', 'glove', char=False, epochs=10, tune=False)

    processPOSTLSTM('long1.5', 'none', 'glove', char=False, epochs=6, tune=False)
    processPOSTLSTM('long1.5', 'class', 'glove', char=False, epochs=6, tune=False)

    #processMLSTM('long1.5', 'none', 'none', char=False, posMode='all', epochs=4)
    #processMLSTM('long1.5', 'class', 'none', char=False, posMode='all', epochs=4)

    #processMLSTM('long1.5', 'none', 'none', char=False, posMode='map', epochs=4)
    #processMLSTM('long1.5', 'class', 'none', char=False, posMode='map', epochs=4)

    #processMLSTM2('long1.5', 'none', 'none', char=False, posMode='all', epochs=10)
