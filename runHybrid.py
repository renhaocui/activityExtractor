import json
import re
import numpy as np
from keras.layers import Dense, LSTM, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from utilities import word2vecReader
from wordsegment import load, segment
from utilities import tokenizer
import sys, evaluation
reload(sys)
sys.setdefaultencoding('utf8')
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


def loadHistData(modelName, char, embedding, histNum=5, pos=False):
    print('Loading...')
    histData = {}
    histFile = open('data/consolidateHistData_' + modelName + '.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        histData[int(data.keys()[0])] = data.values()[0]

    contents = []
    labels = []
    places = []
    days = []
    hours = []
    poss = []
    ids = []
    histContents = {}
    histDayVectors = {}
    histHourVectors = {}
    histPOSLists = {}
    for i in range(histNum):
        histContents[i] = []
        histDayVectors[i] = []
        histHourVectors[i] = []
        histPOSLists[i] = []
    dataFile = open('data/consolidateData_'+modelName+'.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        if data['id'] in histData:
            histTweets = histData[data['id']]
            if len(histTweets) >= histNum:
                contents.append(data['content'].encode('utf-8'))
                labels.append(data['label'])
                places.append(data['place'])
                ids.append(str(data['id']))
                days.append(np.full((tweetLength), data['day'], dtype='int'))
                hours.append(np.full((tweetLength), data['hour'], dtype='int'))
                poss.append(data['pos'].encode('utf-8'))
                for i in range(histNum):
                    histContents[i].append(histTweets[i]['content'].encode('utf-8'))
                    histPOSLists[i].append(histTweets[i]['pos'].encode('utf-8'))
                    histDayVectors[i].append(np.full((tweetLength), histTweets[i]['day'], dtype='int'))
                    histHourVectors[i].append(np.full((tweetLength), histTweets[i]['hour'], dtype='int'))

    for i in range(histNum):
        histDayVectors[i] = np.array(histDayVectors[i])
        histHourVectors[i] = np.array(histHourVectors[i])
    days = np.array(days)
    hours = np.array(hours)
    places = np.array(places)
    ids = np.array(ids)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    totalList = contents[:]
    for i in range(histNum):
        totalList += histContents[i]
    tk.fit_on_texts(totalList)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    histTweetVectors = []
    for i in range(histNum):
        histSequence = tk.texts_to_sequences(histContents[i])
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors.append(tempVector)

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
        return ids, labels, places, contents, days, hours, tweetVector, histTweetVectors, histDayVectors, histHourVectors, embMatrix, word_index
    else:
        posVocabSize = 25
        tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
        totalPOSList = poss[:]
        for i in range(histNum):
            totalPOSList += histPOSLists[i]
        tkPOS.fit_on_texts(totalPOSList)
        posSequences = tkPOS.texts_to_sequences(poss)
        posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

        histPOSVectors = []
        for i in range(histNum):
            histPOSSequences = tkPOS.texts_to_sequences(histPOSLists[i])
            histPOSVector = sequence.pad_sequences(histPOSSequences, maxlen=tweetLength, truncating='post', padding='post')
            histPOSVectors.append(histPOSVector)

        return ids, labels, places, contents, days, hours, poss, tweetVector, posVector, histTweetVectors, histDayVectors, histHourVectors, histPOSVectors, posVocabSize, embMatrix, word_index


def processHistLSTM_contextT(modelName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, tune=False):
    ids, labels, places, contents, dayVector, hourVector, tweetVector, histTweetVectors, histDayVector, histHourVector, embMatrix, word_index = loadHistData(modelName, char, embedding, histNum=histNum, pos=False)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    if tune:
        verbose = 2
    else:
        verbose = 0
    print('training...')

    eval = evaluation.evalMetrics(labelNum)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            shared_embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
            embedding_tweet = shared_embedding_tweet(input_tweet)
        else:
            shared_embedding_tweet = Embedding(vocabSize, embeddingVectorLength)
            embedding_tweet = shared_embedding_tweet(input_tweet)

        input_day = Input(batch_shape=(batch_size, tweetLength,))
        input_hour = Input(batch_shape=(batch_size, tweetLength,))
        shared_embedding_day = Embedding(20, embeddingPOSVectorLength)
        shared_embedding_hour = Embedding(20, embeddingPOSVectorLength)
        embedding_day = shared_embedding_day(input_day)
        embedding_hour = shared_embedding_hour(input_hour)

        comb = concatenate([embedding_tweet, embedding_day, embedding_hour])
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(comb)

        conList = [tweet_lstm]
        inputList = [input_tweet, input_day, input_hour]
        for i in range(histNum):
            input_hist = Input(batch_shape=(batch_size, posEmbLength,))
            embedding_hist_temp = shared_embedding_tweet(input_hist)
            input_day_temp = Input(batch_shape=(batch_size, tweetLength,))
            input_hour_temp = Input(batch_shape=(batch_size, tweetLength,))
            embedding_day_temp = shared_embedding_day(input_day_temp)
            embedding_hour_temp = shared_embedding_hour(input_hour_temp)
            comb_temp = concatenate([embedding_hist_temp, embedding_day_temp, embedding_hour_temp])
            lstm_temp = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(comb_temp)

            conList.append(lstm_temp)
            inputList += [input_hist, input_day_temp, input_hour_temp]

        comb_total = concatenate(conList)
        output = Dense(labelNum, activation='softmax', name='output')(comb_total)
        model = Model(inputs=inputList, outputs=output)
        #print(model.summary())
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
        places_test = places[test_index]
        ids_test = ids[test_index]
        histTweetVector_train = []
        histTweetVector_test = []
        histDayVector_train = []
        histHourVector_train = []
        histDayVector_test = []
        histHourVector_test = []

        for i in range(histNum):
            histTweetVector_train.append(histTweetVectors[i][train_index])
            histTweetVector_test.append(histTweetVectors[i][test_index])
            histDayVector_train.append(histDayVector[i][train_index])
            histHourVector_train.append(histHourVector[i][train_index])
            histHourVector_test.append(histHourVector[i][test_index])
            histDayVector_test.append(histDayVector[i][test_index])

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            day_train = day_train[:-(len(day_train) % batch_size)]
            hour_train = hour_train[:-(len(hour_train) % batch_size)]
            for i in range(histNum):
                histTweetVector_train[i] = histTweetVector_train[i][:-(len(histTweetVector_train[i]) % batch_size)]
                histDayVector_train[i] = histDayVector_train[i][:-(len(histDayVector_train[i]) % batch_size)]
                histHourVector_train[i] = histHourVector_train[i][:-(len(histHourVector_train[i]) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            ids_test = ids_test[:-(len(ids_test) % batch_size)]
            day_test = day_test[:-(len(day_test) % batch_size)]
            hour_test = hour_test[:-(len(hour_test) % batch_size)]
            for i in range(histNum):
                histTweetVector_test[i] = histTweetVector_test[i][:-(len(histTweetVector_test[i]) % batch_size)]
                histDayVector_test[i] = histDayVector_test[i][:-(len(histDayVector_test[i]) % batch_size)]
                histHourVector_test[i] = histHourVector_test[i][:-(len(histHourVector_test[i]) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        trainList = [tweet_train, day_train, hour_train]
        testList = [tweet_test, day_test, hour_test]
        for i in range(histNum):
            trainList += [histTweetVector_train[i], histDayVector_train[i], histHourVector_train[i]]
            testList += [histTweetVector_test[i], histDayVector_test[i], histHourVector_test[i]]

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        tuneFile.write('Hist Num: ' + str(histNum) + '\n')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index + 1) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            sampleFile.write(ids_test[index] + '\t' + contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\t' + places_test[index] + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        eval.addEval(scores[1], labels_test, predLabels)
        if tune:
            break

    score, scoreSTD = eval.getScore()
    precision, preSTD = eval.getPrecision()
    recall, recSTD = eval.getRecall()
    f1, f1STD = eval.getF1()
    conMatrix = eval.getConMatrix()
    resultFile = open('result/C-Hist-Context-T-LSTM_.' + modelName + '_' + balancedWeight + '.result', 'a')
    confusionFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHistLSTM_contextPOS(modelName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, tune=False):
    ids, labels, places, contents, dayVector, hourVector, posList, tweetVector, posVector, histTweetVectors, histDayVector, histHourVector, histPOSVectors, posVocabSize, embMatrix, word_index = loadHistData(modelName, char, embedding,
                                                                                                                                                        histNum=histNum, pos=True)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    if tune:
        verbose = 2
    else:
        verbose = 0
    print('training...')
    eval = evaluation.evalMetrics(labelNum)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            shared_embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
            embedding_tweet = shared_embedding_tweet(input_tweet)
        else:
            shared_embedding_tweet = Embedding(vocabSize, embeddingVectorLength)
            embedding_tweet = shared_embedding_tweet(input_tweet)

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))

        shared_embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)
        embedding_pos = shared_embedding_pos(input_pos)

        comb = concatenate([embedding_tweet, embedding_pos])
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(comb)

        conList = [tweet_lstm]
        inputList = [input_tweet, input_pos]
        for i in range(histNum):
            input_hist = Input(batch_shape=(batch_size, tweetLength,))
            input_pos_temp = Input(batch_shape=(batch_size, posEmbLength,))
            embedding_hist_temp = shared_embedding_tweet(input_hist)
            embedding_pos_temp = shared_embedding_pos(input_pos_temp)
            comb_temp = concatenate([embedding_hist_temp, embedding_pos_temp])
            lstm_temp = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(comb_temp)

            conList.append(lstm_temp)
            inputList += [input_hist, input_pos_temp]

        comb_total = concatenate(conList)
        output = Dense(labelNum, activation='softmax', name='output')(comb_total)
        model = Model(inputs=inputList, outputs=output)
        #print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        pos_train = posVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        pos_test = posVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]
        ids_test = ids[test_index]
        histTweetVector_train = []
        histPOSVector_train = []
        histTweetVector_test = []
        histPOSVector_test = []

        for i in range(histNum):
            histTweetVector_train.append(histTweetVectors[i][train_index])
            histPOSVector_train.append(histPOSVectors[i][train_index])
            histTweetVector_test.append(histTweetVectors[i][test_index])
            histPOSVector_test.append(histPOSVectors[i][test_index])

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            pos_train = pos_train[:-(len(pos_train) % batch_size)]
            for i in range(histNum):
                histTweetVector_train[i] = histTweetVector_train[i][:-(len(histTweetVector_train[i]) % batch_size)]
                histPOSVector_train[i] = histPOSVector_train[i][:-(len(histPOSVector_train[i]) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            ids_test = ids_test[:-(len(ids_test) % batch_size)]
            pos_test = pos_test[:-(len(pos_test) % batch_size)]
            for i in range(histNum):
                histTweetVector_test[i] = histTweetVector_test[i][:-(len(histTweetVector_test[i]) % batch_size)]
                histPOSVector_test[i] = histPOSVector_test[i][:-(len(histPOSVector_test[i]) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        trainList = [tweet_train, pos_train]
        testList = [tweet_test, pos_test]
        for i in range(histNum):
            trainList += [histTweetVector_train[i], histPOSVector_train[i]]
            testList += [histTweetVector_test[i], histPOSVector_test[i]]

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        tuneFile.write('Hist Num: ' + str(histNum) + '\n')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index + 1) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            sampleFile.write(ids_test[index] + '\t' + contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\t' + places_test[index] + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        eval.addEval(scores[1], labels_test, predLabels)
        if tune:
            break

    score, scoreSTD = eval.getScore()
    precision, preSTD = eval.getPrecision()
    recall, recSTD = eval.getRecall()
    f1, f1STD = eval.getF1()
    conMatrix = eval.getConMatrix()
    resultFile = open('result/C-Hist-Context-POS-LSTM_.' + modelName + '_' + balancedWeight + '.result', 'a')
    confusionFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHistLSTM_contextPOST(modelName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, tune=False):
    ids, labels, places, contents, dayVector, hourVector, posList, tweetVector, posVector, histTweetVectors, histDayVector, histHourVector, histPOSVectors, posVocabSize, embMatrix, word_index = loadHistData(
        modelName, char, embedding,
        histNum=histNum, pos=True)

    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    if tune:
        verbose = 2
    else:
        verbose = 0
    print('training...')

    eval = evaluation.evalMetrics(labelNum)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            shared_embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
            embedding_tweet = shared_embedding_tweet(input_tweet)
        else:
            shared_embedding_tweet = Embedding(vocabSize, embeddingVectorLength)
            embedding_tweet = shared_embedding_tweet(input_tweet)

        input_day = Input(batch_shape=(batch_size, tweetLength,))
        input_hour = Input(batch_shape=(batch_size, tweetLength,))
        input_pos = Input(batch_shape=(batch_size, posEmbLength,))

        shared_embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)
        shared_embedding_day = Embedding(20, embeddingPOSVectorLength)
        shared_embedding_hour = Embedding(20, embeddingPOSVectorLength)
        embedding_day = shared_embedding_day(input_day)
        embedding_hour = shared_embedding_hour(input_hour)
        embedding_pos = shared_embedding_pos(input_pos)

        comb = concatenate([embedding_tweet, embedding_day, embedding_hour, embedding_pos])
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(comb)

        conList = [tweet_lstm]
        inputList = [input_tweet, input_day, input_hour, input_pos]
        for i in range(histNum):
            input_hist = Input(batch_shape=(batch_size, tweetLength,))
            input_day_temp = Input(batch_shape=(batch_size, tweetLength,))
            input_hour_temp = Input(batch_shape=(batch_size, tweetLength,))
            input_pos_temp = Input(batch_shape=(batch_size, posEmbLength,))
            embedding_hist_temp = shared_embedding_tweet(input_hist)
            embedding_day_temp = shared_embedding_day(input_day_temp)
            embedding_hour_temp = shared_embedding_hour(input_hour_temp)
            embedding_pos_temp = shared_embedding_pos(input_pos_temp)
            comb_temp = concatenate([embedding_hist_temp, embedding_day_temp, embedding_hour_temp, embedding_pos_temp])
            lstm_temp = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(comb_temp)

            conList.append(lstm_temp)
            inputList += [input_hist, input_day_temp, input_hour_temp, input_pos_temp]

        comb_total = concatenate(conList)
        output = Dense(labelNum, activation='softmax', name='output')(comb_total)
        model = Model(inputs=inputList, outputs=output)
        #print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        day_train = dayVector[train_index]
        hour_train = hourVector[train_index]
        pos_train = posVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        day_test = dayVector[test_index]
        hour_test = hourVector[test_index]
        pos_test = posVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]
        ids_test = ids[test_index]
        histTweetVector_train = []
        histDayVector_train = []
        histHourVector_train = []
        histPOSVector_train = []
        histTweetVector_test = []
        histDayVector_test = []
        histHourVector_test = []
        histPOSVector_test = []

        for i in range(histNum):
            histTweetVector_train.append(histTweetVectors[i][train_index])
            histDayVector_train.append(histDayVector[i][train_index])
            histHourVector_train.append(histHourVector[i][train_index])
            histPOSVector_train.append(histPOSVectors[i][train_index])
            histTweetVector_test.append(histTweetVectors[i][test_index])
            histDayVector_test.append(histDayVector[i][test_index])
            histHourVector_test.append(histHourVector[i][test_index])
            histPOSVector_test.append(histPOSVectors[i][test_index])

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            day_train = day_train[:-(len(day_train) % batch_size)]
            hour_train = hour_train[:-(len(hour_train) % batch_size)]
            pos_train = pos_train[:-(len(pos_train) % batch_size)]
            for i in range(histNum):
                histTweetVector_train[i] = histTweetVector_train[i][:-(len(histTweetVector_train[i]) % batch_size)]
                histDayVector_train[i] = histDayVector_train[i][:-(len(histDayVector_train[i]) % batch_size)]
                histHourVector_train[i] = histHourVector_train[i][:-(len(histHourVector_train[i]) % batch_size)]
                histPOSVector_train[i] = histPOSVector_train[i][:-(len(histPOSVector_train[i]) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            ids_test = ids_test[:-(len(ids_test) % batch_size)]
            day_test = day_test[:-(len(day_test) % batch_size)]
            hour_test = hour_test[:-(len(hour_test) % batch_size)]
            pos_test = pos_test[:-(len(pos_test) % batch_size)]
            for i in range(histNum):
                histTweetVector_test[i] = histTweetVector_test[i][:-(len(histTweetVector_test[i]) % batch_size)]
                histDayVector_test[i] = histDayVector_test[i][:-(len(histDayVector_test[i]) % batch_size)]
                histHourVector_test[i] = histHourVector_test[i][:-(len(histHourVector_test[i]) % batch_size)]
                histPOSVector_test[i] = histPOSVector_test[i][:-(len(histPOSVector_test[i]) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        trainList = [tweet_train, day_train, hour_train, pos_train]
        testList = [tweet_test, day_test, hour_test, pos_test]
        for i in range(histNum):
            trainList += [histTweetVector_train[i], histDayVector_train[i], histHourVector_train[i], histPOSVector_train[i]]
            testList += [histTweetVector_test[i], histDayVector_test[i], histHourVector_test[i], histPOSVector_test[i]]

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.tune', 'a')
        tuneFile.write('Hist Num: ' + str(histNum) + '\n')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index + 1) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            sampleFile.write(ids_test[index] + '\t' + contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\t' + places_test[index] + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        eval.addEval(scores[1], labels_test, predLabels)
        if tune:
            break

    score, scoreSTD = eval.getScore()
    precision, preSTD = eval.getPrecision()
    recall, recSTD = eval.getRecall()
    f1, f1STD = eval.getF1()
    conMatrix = eval.getConMatrix()
    resultFile = open('result/C-Hist-Context-POST-LSTM_.' + modelName + '_' + balancedWeight + '.result', 'a')
    confusionFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


if __name__ == "__main__":
    processHistLSTM_contextT('long1.5', 'none', 'glove', char=False, histNum=5, epochs=10, tune=False)
    processHistLSTM_contextT('long1.5', 'class', 'glove', char=False, histNum=5, epochs=27, tune=False)

    #processHistLSTM_contextPOS('long1.5', 'none', 'glove', char=False, histNum=5, epochs=25, tune=False)
    #processHistLSTM_contextPOS('long1.5', 'class', 'glove', char=False, histNum=3, epochs=6, tune=False)

    #processHistLSTM_contextPOST('long1.5', 'none', 'glove', char=False, histNum=3, epochs=6, tune=False)
    #processHistLSTM_contextPOST('long1.5', 'class', 'glove', char=False, histNum=5, epochs=10, tune=False)

