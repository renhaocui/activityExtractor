import json, re, time, datetime, sys
import numpy as np
from keras.layers import Dense, LSTM, Input, concatenate, Lambda
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
from utilities import tokenizer, evaluation
from keras_self_attention import SeqSelfAttention
reload(sys)
sys.setdefaultencoding('utf8')

vocabSize = 10000
tweetLength = 25
posEmbLength = 25
embeddingVectorLength = 200
embeddingPOSVectorLength = 20
charLengthLimit = 20
batch_size = 100
load()

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


def loadData(modelName, char, embedding, pos=False, dev=False):
    print('Loading...')
    contents_train = []
    contents_val = []
    labels_train = []
    labels_val = []
    places_train = []
    places_val = []
    days_train = []
    days_val = []
    hours_train = []
    hours_val = []
    poss_train = []
    poss_val = []
    ids_train = []
    ids_val = []
    trainFile = open('data/consolidateData_'+modelName+'_train.json', 'r')
    if dev:
        valFile = open('data/consolidateData_'+modelName+'_test.json', 'r')
    else:
        valFile = open('data/consolidateData_'+modelName+'_dev.json', 'r')

    for line in trainFile:
        data = json.loads(line.strip())
        contents_train.append(data['content'].encode('utf-8'))
        labels_train.append(data['label'])
        places_train.append(data['place'])
        days_train.append(np.full(tweetLength, data['day'], dtype='int'))
        hours_train.append(np.full(tweetLength, data['hour'], dtype='int'))
        poss_train.append(data['pos'].encode('utf-8'))
        ids_train.append(str(data['id']))
    places_train = np.array(places_train)
    ids_train = np.array(ids_train)
    labels_train = np.array(labels_train)

    for line in valFile:
        data = json.loads(line.strip())
        contents_val.append(data['content'].encode('utf-8'))
        labels_val.append(data['label'])
        places_val.append(data['place'])
        days_val.append(np.full(tweetLength, data['day'], dtype='int'))
        hours_val.append(np.full(tweetLength, data['hour'], dtype='int'))
        poss_val.append(data['pos'].encode('utf-8'))
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

    days_train = np.array(days_train)
    days_val = np.array(days_val)
    hours_train = np.array(hours_train)
    hours_val = np.array(hours_val)

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
        return ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, tweetVector_train, tweetVector_val, embMatrix, word_index
    else:
        posVocabSize = 25
        tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
        tkPOS.fit_on_texts(poss_train + poss_val)
        posSequences_train = tkPOS.texts_to_sequences(poss_train)
        posVector_train = sequence.pad_sequences(posSequences_train, maxlen=tweetLength, truncating='post', padding='post')
        posSequences_val = tkPOS.texts_to_sequences(poss_val)
        posVector_val = sequence.pad_sequences(posSequences_val, maxlen=tweetLength, truncating='post', padding='post')

        return ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, poss_train, poss_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, posVocabSize, embMatrix, word_index


def loadHistData(modelName, histName, char, embedding, histNum=5, pos=False, dev=False):
    print('Loading...')
    histData = {}
    histFile = open('data/consolidateHistData_' + histName + '.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        histData[int(data.keys()[0])] = data.values()[0]
    histFile.close()

    histContents_train = {}
    histDayVectors_train = {}
    histHourVectors_train = {}
    histPOSLists_train = {}
    histContents_val = {}
    histDayVectors_val = {}
    histHourVectors_val = {}
    histPOSLists_val = {}
    for i in range(histNum):
        histContents_train[i] = []
        histDayVectors_train[i] = []
        histHourVectors_train[i] = []
        histPOSLists_train[i] = []
        histContents_val[i] = []
        histDayVectors_val[i] = []
        histHourVectors_val[i] = []
        histPOSLists_val[i] = []

    contents_train = []
    contents_val = []
    labels_train = []
    labels_val = []
    places_train = []
    places_val = []
    days_train = []
    days_val = []
    hours_train = []
    hours_val = []
    poss_train = []
    poss_val = []
    ids_train = []
    ids_val = []
    trainFile = open('data/consolidateData_' + modelName + '_train.json', 'r')
    if dev:
        valFile = open('data/consolidateData_' + modelName + '_test.json', 'r')
    else:
        valFile = open('data/consolidateData_' + modelName + '_dev.json', 'r')

    for line in trainFile:
        data = json.loads(line.strip())
        if data['id'] in histData:
            histTweets = histData[data['id']]
            if len(histTweets) >= 5:
                contents_train.append(data['content'].encode('utf-8'))
                labels_train.append(data['label'])
                places_train.append(data['place'])
                ids_train.append(str(data['id']))
                days_train.append(np.full((tweetLength), data['day'], dtype='int'))
                hours_train.append(np.full((tweetLength), data['hour'], dtype='int'))
                poss_train.append(data['pos'].encode('utf-8'))
                for i in range(histNum):
                    histContents_train[i].append(histTweets[i]['content'].encode('utf-8'))
                    histPOSLists_train[i].append(histTweets[i]['pos'].encode('utf-8'))
                    histDayVectors_train[i].append(np.full((tweetLength), histTweets[i]['day'], dtype='int'))
                    histHourVectors_train[i].append(np.full((tweetLength), histTweets[i]['hour'], dtype='int'))

    for i in range(histNum):
        histDayVectors_train[i] = np.array(histDayVectors_train[i])
        histHourVectors_train[i] = np.array(histHourVectors_train[i])
    days_train = np.array(days_train)
    hours_train = np.array(hours_train)
    places_train = np.array(places_train)
    ids_train = np.array(ids_train)

    for line in valFile:
        data = json.loads(line.strip())
        if data['id'] in histData:
            histTweets = histData[data['id']]
            if len(histTweets) >= histNum:
                contents_val.append(data['content'].encode('utf-8'))
                labels_val.append(data['label'])
                places_val.append(data['place'])
                ids_val.append(str(data['id']))
                days_val.append(np.full((tweetLength), data['day'], dtype='int'))
                hours_val.append(np.full((tweetLength), data['hour'], dtype='int'))
                poss_val.append(data['pos'].encode('utf-8'))
                for i in range(histNum):
                    histContents_val[i].append(histTweets[i]['content'].encode('utf-8'))
                    histPOSLists_val[i].append(histTweets[i]['pos'].encode('utf-8'))
                    histDayVectors_val[i].append(np.full((tweetLength), histTweets[i]['day'], dtype='int'))
                    histHourVectors_val[i].append(np.full((tweetLength), histTweets[i]['hour'], dtype='int'))

    for i in range(histNum):
        histDayVectors_val[i] = np.array(histDayVectors_val[i])
        histHourVectors_val[i] = np.array(histHourVectors_val[i])
    days_val = np.array(days_val)
    hours_val = np.array(hours_val)
    places_val = np.array(places_val)
    ids_val = np.array(ids_val)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    totalList = contents_train + contents_val
    for i in range(histNum):
        totalList += histContents_train[i]
        totalList += histContents_val[i]
    tk.fit_on_texts(totalList)
    tweetSequences_train = tk.texts_to_sequences(contents_train)
    tweetVector_train = sequence.pad_sequences(tweetSequences_train, maxlen=tweetLength, truncating='post', padding='post')
    tweetSequences_val = tk.texts_to_sequences(contents_val)
    tweetVector_val = sequence.pad_sequences(tweetSequences_val, maxlen=tweetLength, truncating='post', padding='post')

    histTweetVectors_train = []
    histTweetVectors_val = []
    for i in range(histNum):
        histSequence_train = tk.texts_to_sequences(histContents_train[i])
        tempVector_train = sequence.pad_sequences(histSequence_train, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors_train.append(tempVector_train)
        histSequence_val = tk.texts_to_sequences(histContents_val[i])
        tempVector_val = sequence.pad_sequences(histSequence_val, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors_val.append(tempVector_val)

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
        return ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, tweetVector_train, tweetVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, histDayVectors_val, histHourVectors_train, histHourVectors_val, embMatrix, word_index
    else:
        posVocabSize = 25
        tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
        totalPOSList = poss_train + poss_val
        for i in range(histNum):
            totalPOSList += histPOSLists_train[i]
            totalPOSList += histPOSLists_val[i]
        tkPOS.fit_on_texts(totalPOSList)

        posSequences_train = tkPOS.texts_to_sequences(poss_train)
        posVector_train = sequence.pad_sequences(posSequences_train, maxlen=tweetLength, truncating='post', padding='post')
        posSequences_val = tkPOS.texts_to_sequences(poss_val)
        posVector_val = sequence.pad_sequences(posSequences_val, maxlen=tweetLength, truncating='post', padding='post')

        histPOSVectors_train = []
        histPOSVectors_val = []
        for i in range(histNum):
            histPOSSequences_train = tkPOS.texts_to_sequences(histPOSLists_train[i])
            histPOSVector_train = sequence.pad_sequences(histPOSSequences_train, maxlen=tweetLength, truncating='post', padding='post')
            histPOSVectors_train.append(histPOSVector_train)
            histPOSSequences_val = tkPOS.texts_to_sequences(histPOSLists_val[i])
            histPOSVector_val = sequence.pad_sequences(histPOSSequences_val, maxlen=tweetLength, truncating='post', padding='post')
            histPOSVectors_val.append(histPOSVector_val)

        return ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, poss_train, poss_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, histDayVectors_val, histHourVectors_train, histHourVectors_val, histPOSVectors_train, histPOSVectors_val, posVocabSize, embMatrix, word_index


def loadHistData2(modelName, char, embedding, histNum=5, pos=False):
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


def processContextPOSLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=7, dev=False):
    resultName = 'result/Context-POSLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, dayList_train, dayList_val, \
    hourList_train, hourList_val, posList_train, posList_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, posVocabSize, embMatrix, word_index = loadData(
        modelName, char, embedding, pos=True, dev=dev)

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
    print('training...')
    if dev:
        verbose = 2
    else:
        verbose = 0

    eval = evaluation.evalMetrics(labelNum)

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)

    input_pos = Input(batch_shape=(batch_size, posEmbLength,))
    embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)

    embedding_comb = concatenate([embedding_tweet, embedding_pos])
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
    output = Dense(labelNum, activation='softmax', name='output')(lstm)
    model = Model(inputs=[input_tweet, input_pos], outputs=output)
    #print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        posVector_val = posVector_val[:-(len(posVector_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit([tweetVector_train, posVector_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit([tweetVector_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, posVector_val], labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit([tweetVector_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, posVector_val], labelVector_val), batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']
    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate([tweetVector_val, posVector_val], labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict([tweetVector_val, posVector_val], batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processContextTLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    resultName = 'result/Context-TLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, dayList_train, dayList_val, \
    hourList_train, hourList_val, tweetVector_train, tweetVector_val, embMatrix, word_index = loadData(modelName, char, embedding, pos=False, dev=dev)

    trainSize = len(ids_train)
    valSize = len(ids_val)
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
    print('training...')
    if dev:
        verbose = 2
    else:
        verbose = 0
    eval = evaluation.evalMetrics(labelNum)

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)

    input_day = Input(batch_shape=(batch_size, tweetLength,))
    input_hour = Input(batch_shape=(batch_size, tweetLength,))
    embedding_day = Embedding(20, embeddingPOSVectorLength)(input_day)
    embedding_hour = Embedding(20, embeddingPOSVectorLength)(input_hour)

    embedding_comb = concatenate([embedding_tweet, embedding_day, embedding_hour])
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
    output = Dense(labelNum, activation='softmax', name='output')(lstm)
    model = Model(inputs=[input_tweet, input_day, input_hour], outputs=output)
    #print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        hourList_train = hourList_train[:-(len(hourList_train) % batch_size)]
        dayList_train = dayList_train[:-(len(dayList_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        hourList_val = hourList_val[:-(len(hourList_val) % batch_size)]
        dayList_val = dayList_val[:-(len(dayList_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit([tweetVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit([tweetVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, dayList_val, hourList_val], labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit([tweetVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, dayList_val, hourList_val], labelVector_val), batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']
    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate([tweetVector_val, dayList_val, hourList_val], labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict([tweetVector_val, dayList_val, hourList_val], batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processContextPOSTLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    resultName = 'result/Context-POSTLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, dayList_train, dayList_val, \
    hourList_train, hourList_val, posList_train, posList_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, posVocabSize, embMatrix, word_index = loadData(
        modelName, char, embedding, pos=True, dev=dev)

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
    print('training...')
    if dev:
        verbose = 2
    else:
        verbose = 0
    eval = evaluation.evalMetrics(labelNum)

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)

    input_pos = Input(batch_shape=(batch_size, posEmbLength,))
    embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)

    input_day = Input(batch_shape=(batch_size, tweetLength,))
    input_hour = Input(batch_shape=(batch_size, tweetLength,))
    embedding_day = Embedding(20, embeddingPOSVectorLength)(input_day)
    embedding_hour = Embedding(20, embeddingPOSVectorLength)(input_hour)

    embedding_comb = concatenate([embedding_tweet, embedding_pos, embedding_day, embedding_hour])
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
    output = Dense(labelNum, activation='softmax', name='output')(lstm)
    model = Model(inputs=[input_tweet, input_pos, input_day, input_hour], outputs=output)
    #print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        hourList_train = hourList_train[:-(len(hourList_train) % batch_size)]
        dayList_train = dayList_train[:-(len(dayList_train) % batch_size)]
        posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        hourList_val = hourList_val[:-(len(hourList_val) % batch_size)]
        dayList_val = dayList_val[:-(len(dayList_val) % batch_size)]
        posVector_val = posVector_val[:-(len(posVector_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit([tweetVector_train, posVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit([tweetVector_train, posVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, posVector_val, dayList_val, hourList_val], labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit([tweetVector_train, posVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, posVector_val, dayList_val, hourList_val], labelVector_val), batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']
    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate([tweetVector_val, posVector_val, dayList_val, hourList_val], labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict([tweetVector_val, posVector_val, dayList_val, hourList_val], batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processContextAttPOSTLSTM(modelName, balancedWeight='None', embedding='None', char=False, epochs=4, dev=False):
    resultName = 'result/Context-Att-POSTLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, dayList_train, dayList_val, \
    hourList_train, hourList_val, posList_train, posList_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, posVocabSize, embMatrix, word_index = loadData(
        modelName, char, embedding, pos=True, dev=dev)

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
    print('training...')
    if dev:
        verbose = 2
    else:
        verbose = 0
    eval = evaluation.evalMetrics(labelNum)

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)(input_tweet)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)

    input_pos = Input(batch_shape=(batch_size, posEmbLength,))
    embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)

    input_day = Input(batch_shape=(batch_size, tweetLength,))
    input_hour = Input(batch_shape=(batch_size, tweetLength,))
    embedding_day = Embedding(20, embeddingPOSVectorLength)(input_day)
    embedding_hour = Embedding(20, embeddingPOSVectorLength)(input_hour)

    embedding_comb = concatenate([embedding_tweet, embedding_pos, embedding_day, embedding_hour])
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_comb)
    self_attention = SeqSelfAttention(attention_activation='sigmoid')(lstm)
    last_timestep = Lambda(lambda x: x[:, -1, :])(self_attention)
    output = Dense(labelNum, activation='softmax', name='output')(last_timestep)
    model = Model(inputs=[input_tweet, input_pos, input_day, input_hour], outputs=output)
    #print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        hourList_train = hourList_train[:-(len(hourList_train) % batch_size)]
        dayList_train = dayList_train[:-(len(dayList_train) % batch_size)]
        posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        hourList_val = hourList_val[:-(len(hourList_val) % batch_size)]
        dayList_val = dayList_val[:-(len(dayList_val) % batch_size)]
        posVector_val = posVector_val[:-(len(posVector_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit([tweetVector_train, posVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit([tweetVector_train, posVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, posVector_val, dayList_val, hourList_val], labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit([tweetVector_train, posVector_train, dayList_train, hourList_train], labelVector_train, epochs=epochs, validation_data=([tweetVector_val, posVector_val, dayList_val, hourList_val], labelVector_val), batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']
    tuneFile = open(resultName + '.tune', 'a')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate([tweetVector_val, posVector_val, dayList_val, hourList_val], labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict([tweetVector_val, posVector_val, dayList_val, hourList_val], batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processContextHistLSTM(modelName, histName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, dev=False):
    resultName = 'result/Context-HistLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, \
    tweetVector_train, tweetVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, histDayVectors_val, histHourVectors_train, histHourVectors_val, embMatrix, word_index = loadHistData(
        modelName, histName, char, embedding, histNum=histNum, pos=False, dev=dev)

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

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        shared_embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
        embedding_tweet = shared_embedding(input_tweet)
    else:
        shared_embedding = Embedding(vocabSize, embeddingVectorLength)
        embedding_tweet = shared_embedding(input_tweet)

    conList = [embedding_tweet]
    inputList = [input_tweet]
    for i in range(histNum):
        input_hist = Input(batch_shape=(batch_size, tweetLength,))
        embedding_hist = shared_embedding(input_hist)
        conList.append(embedding_hist)
        inputList.append(input_hist)

    embedding_comb = concatenate(conList)
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
    output = Dense(labelNum, activation='softmax', name='output')(lstm)
    model = Model(inputs=inputList, outputs=output)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        for i in range(histNum):
            histTweetVectors_train[i] = histTweetVectors_train[i][:-(len(histTweetVectors_train[i]) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        for i in range(histNum):
            histTweetVectors_val[i] = histTweetVectors_val[i][:-(len(histTweetVectors_val[i]) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    dataVector_train = [tweetVector_train]
    dataVector_val = [tweetVector_val]
    for histVector in histTweetVectors_train:
        dataVector_train.append(histVector)
    for histVector in histTweetVectors_val:
        dataVector_val.append(histVector)

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    tuneFile.write('Hist Num: '+str(histNum)+'\n')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index+1) + '\t' + str(loss)+'\t'+str(accuracyHist[index])+'\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(dataVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict(dataVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processContextAttHistLSTM(modelName, histName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, dev=False):
    resultName = 'result/Context-Att-HistLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, \
    tweetVector_train, tweetVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, histDayVectors_val, histHourVectors_train, histHourVectors_val, embMatrix, word_index = loadHistData(
        modelName, histName, char, embedding, histNum=histNum, pos=False, dev=dev)

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

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        shared_embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
        embedding_tweet = shared_embedding(input_tweet)
    else:
        shared_embedding = Embedding(vocabSize, embeddingVectorLength)
        embedding_tweet = shared_embedding(input_tweet)

    conList = [embedding_tweet]
    inputList = [input_tweet]
    for i in range(histNum):
        input_hist = Input(batch_shape=(batch_size, tweetLength,))
        embedding_hist = shared_embedding(input_hist)
        conList.append(embedding_hist)
        inputList.append(input_hist)

    embedding_comb = concatenate(conList)
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_comb)
    self_attention = SeqSelfAttention(attention_activation='sigmoid')(lstm)
    last_timestep = Lambda(lambda x: x[:, -1, :])(self_attention)
    output = Dense(labelNum, activation='softmax', name='output')(last_timestep)
    model = Model(inputs=inputList, outputs=output)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        for i in range(histNum):
            histTweetVectors_train[i] = histTweetVectors_train[i][:-(len(histTweetVectors_train[i]) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        for i in range(histNum):
            histTweetVectors_val[i] = histTweetVectors_val[i][:-(len(histTweetVectors_val[i]) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    dataVector_train = [tweetVector_train]
    dataVector_val = [tweetVector_val]
    for histVector in histTweetVectors_train:
        dataVector_train.append(histVector)
    for histVector in histTweetVectors_val:
        dataVector_val.append(histVector)

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    tuneFile.write('Hist Num: '+str(histNum)+'\n')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index+1) + '\t' + str(loss)+'\t'+str(accuracyHist[index])+'\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(dataVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict(dataVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processHistLSTM_period(modelName, balancedWeight='None', embedding='None', periodNum=5, epochs=4, tune=False):
    print('Loading...')
    resultName = 'result/C-HistLSTM_period_' + modelName + '_' + balancedWeight
    emptyVector = []
    for i in range(tweetLength):
        emptyVector.append(0)
    maxHistNum = 50
    minHistNum = 1
    histDataSet = {}
    histFile = open('data/consolidateHistData_' + modelName + '_max.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        histDataSet[int(data.keys()[0])] = data.values()[0]
    histFile.close()

    totalContents = []
    contents = []
    labels = []
    places = []
    ids = []
    histData = []
    dataFile = open('data/consolidateData_' + modelName + '_CreatedAt.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        if data['id'] in histDataSet:
            histTweets = histDataSet[data['id']]
            if len(histTweets) >= minHistNum:
                tempData = []
                totalContents.append(data['content'].encode('utf-8'))
                contents.append(data['content'].encode('utf-8'))
                labels.append(data['label'])
                places.append(data['place'])
                ids.append(str(data['id']))
                timeTemp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                createdTimestamp = time.mktime(datetime.datetime.strptime(timeTemp, '%Y-%m-%d %H:%M:%S').timetuple())
                for histTweet in reversed(histTweets):
                    timeTemp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(histTweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                    histCreatedTimestamp = time.mktime(datetime.datetime.strptime(timeTemp, '%Y-%m-%d %H:%M:%S').timetuple())
                    if (createdTimestamp - histCreatedTimestamp) < periodNum * 8 * 3600:
                        totalContents.append(histTweet['content'].encode('utf-8'))
                        tempData.append(histTweet['content'].encode('utf-8'))
                histData.append(tempData)

    places = np.array(places)
    ids = np.array(ids)
    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    tk = Tokenizer(num_words=vocabSize)
    tk.fit_on_texts(totalContents)

    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')
    print tweetVector.shape
    histVector = []
    for tempData in histData:
        tweetSequences = tk.texts_to_sequences(tempData)
        histTweetVectors = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')
        if len(histTweetVectors) == 0:
            histTweetVectors = np.zeros((maxHistNum, tweetLength))
        elif len(histTweetVectors) < maxHistNum:
            for i in range(maxHistNum - len(histTweetVectors)):
                histTweetVectors = np.append(histTweetVectors, [emptyVector], axis=0)
        histVector.append(histTweetVectors)
    histVectors = np.array(histVector)
    # print len(dataVector[0])
    print histVectors.shape

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
            shared_embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
            embedding_tweet = shared_embedding(input_tweet)
        else:
            shared_embedding = Embedding(vocabSize, embeddingVectorLength)
            embedding_tweet = shared_embedding(input_tweet)

        conList = [embedding_tweet]
        inputList = [input_tweet]
        for i in range(maxHistNum):
            input_hist = Input(batch_shape=(batch_size, tweetLength,))
            embedding_hist = shared_embedding(input_hist)
            conList.append(embedding_hist)
            inputList.append(input_hist)

        embedding_comb = concatenate(conList)
        lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
        output = Dense(labelNum, activation='softmax', name='output')(lstm)
        model = Model(inputs=inputList, outputs=output)
        #print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]
        ids_test = ids[test_index]
        histVector_train = histVectors[train_index]
        histVector_test = histVectors[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            histVector_train = histVector_train[:-(len(histVector_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            ids_test = ids_test[:-(len(ids_test) % batch_size)]
            histVector_test = histVector_test[:-(len(histVector_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        histVector_train = np.swapaxes(histVector_train, 0, 1)
        histVector_test = np.swapaxes(histVector_test, 0, 1)
        print histVector_train.shape
        print tweet_train.shape

        trainList = [tweet_train]
        testList = [tweet_test]
        for i in range(maxHistNum):
            trainList.append(histVector_train[i])
            testList.append(histVector_test[i])

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open(resultName + '.tune', 'a')
        tuneFile.write('Period Num: '+str(periodNum)+'\n')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index+1) + '\t' + str(loss)+'\t'+str(accuracyHist[index])+'\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open(resultName + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            sampleFile.write(ids_test[index] + '\t' + contents_test[index] + '\t' + labels_test[index] + '\t' + predLabel + '\t' + places_test[index] + '\n')
            predLabels.append(predLabel)
        sampleFile.close()
        eval.addEval(scores[1], labels_test, predLabels)
        if tune:
            break

    if not tune:
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


def processContextHistPOSTLSTM(modelName, histName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, dev=False):
    resultName = 'result/Context-HistPOSTLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, \
    poss_train, poss_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, \
    histDayVectors_val, histHourVectors_train, histHourVectors_val, histPOSVectors_train, histPOSVectors_val, posVocabSize, embMatrix, word_index = loadHistData(
        modelName, histName, char, embedding, histNum=histNum, pos=True, dev=dev)

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

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        shared_embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
        embedding_tweet = shared_embedding(input_tweet)
    else:
        shared_embedding = Embedding(vocabSize, embeddingVectorLength)
        embedding_tweet = shared_embedding(input_tweet)

    input_pos = Input(batch_shape=(batch_size, posEmbLength,))
    shared_embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)
    embedding_pos = shared_embedding_pos(input_pos)

    input_day = Input(batch_shape=(batch_size, tweetLength,))
    input_hour = Input(batch_shape=(batch_size, tweetLength,))
    shared_embedding_day = Embedding(20, embeddingPOSVectorLength)
    embedding_day = shared_embedding_day(input_day)
    shared_embedding_hour = Embedding(20, embeddingPOSVectorLength)
    embedding_hour = shared_embedding_hour(input_hour)

    conList = [embedding_tweet, embedding_pos, embedding_day, embedding_hour]
    inputList = [input_tweet, input_pos, input_day, input_hour]
    for i in range(histNum):
        input_hist = Input(batch_shape=(batch_size, tweetLength,))
        embedding_hist = shared_embedding(input_hist)
        conList.append(embedding_hist)
        inputList.append(input_hist)

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = shared_embedding_pos(input_pos)
        conList.append(embedding_pos)
        inputList.append(input_pos)

        input_day = Input(batch_shape=(batch_size, tweetLength,))
        input_hour = Input(batch_shape=(batch_size, tweetLength,))
        embedding_day = shared_embedding_day(input_day)
        embedding_hour = shared_embedding_hour(input_hour)
        conList.append(embedding_day)
        conList.append(embedding_hour)
        inputList.append(input_day)
        inputList.append(input_hour)

    embedding_comb = concatenate(conList)
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
    output = Dense(labelNum, activation='softmax', name='output')(lstm)
    model = Model(inputs=inputList, outputs=output)
    #print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        days_train = days_train[:-(len(days_train) % batch_size)]
        hours_train = hours_train[:-(len(hours_train) % batch_size)]
        posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        for i in range(histNum):
            histTweetVectors_train[i] = histTweetVectors_train[i][:-(len(histTweetVectors_train[i]) % batch_size)]
            histDayVectors_train[i] = histDayVectors_train[i][:-(len(histDayVectors_train[i]) % batch_size)]
            histHourVectors_train[i] = histHourVectors_train[i][:-(len(histHourVectors_train[i]) % batch_size)]
            histPOSVectors_train[i] = histPOSVectors_train[i][:-(len(histPOSVectors_train[i]) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        days_val = days_val[:-(len(days_val) % batch_size)]
        hours_val = hours_val[:-(len(hours_val) % batch_size)]
        posVector_val = posVector_val[:-(len(posVector_val) % batch_size)]
        for i in range(histNum):
            histTweetVectors_val[i] = histTweetVectors_val[i][:-(len(histTweetVectors_val[i]) % batch_size)]
            histDayVectors_val[i] = histDayVectors_val[i][:-(len(histDayVectors_val[i]) % batch_size)]
            histHourVectors_val[i] = histHourVectors_val[i][:-(len(histHourVectors_val[i]) % batch_size)]
            histPOSVectors_val[i] = histPOSVectors_val[i][:-(len(histPOSVectors_val[i]) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    dataVector_train = [tweetVector_train, posVector_train, days_train, hours_train]
    dataVector_val = [tweetVector_val, posVector_val, days_val, hours_val]

    for i in range(histNum):
        dataVector_train += [histTweetVectors_train[i], histPOSVectors_train[i], histDayVectors_train[i], histHourVectors_train[i]]
        dataVector_val += [histTweetVectors_val[i], histPOSVectors_val[i], histDayVectors_val[i], histHourVectors_val[i]]

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size,
                                 class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size,
                                 verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    tuneFile.write('Hist Num: ' + str(histNum) + '\n')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index + 1) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(dataVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict(dataVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


def processContextAttHistPOSTLSTM(modelName, histName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=7, dev=False):
    resultName = 'result/ContextAtt-HistPOSTLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, \
    poss_train, poss_val, tweetVector_train, tweetVector_val, posVector_train, posVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, \
    histDayVectors_val, histHourVectors_train, histHourVectors_val, histPOSVectors_train, histPOSVectors_val, posVocabSize, embMatrix, word_index = loadHistData(
        modelName, histName, char, embedding, histNum=histNum, pos=True, dev=dev)

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

    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        shared_embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
        embedding_tweet = shared_embedding(input_tweet)
    else:
        shared_embedding = Embedding(vocabSize, embeddingVectorLength)
        embedding_tweet = shared_embedding(input_tweet)

    input_pos = Input(batch_shape=(batch_size, posEmbLength,))
    shared_embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)
    embedding_pos = shared_embedding_pos(input_pos)

    input_day = Input(batch_shape=(batch_size, tweetLength,))
    input_hour = Input(batch_shape=(batch_size, tweetLength,))
    shared_embedding_day = Embedding(20, embeddingPOSVectorLength)
    embedding_day = shared_embedding_day(input_day)
    shared_embedding_hour = Embedding(20, embeddingPOSVectorLength)
    embedding_hour = shared_embedding_hour(input_hour)

    conList = [embedding_tweet, embedding_pos, embedding_day, embedding_hour]
    inputList = [input_tweet, input_pos, input_day, input_hour]
    for i in range(histNum):
        input_hist = Input(batch_shape=(batch_size, tweetLength,))
        embedding_hist = shared_embedding(input_hist)
        conList.append(embedding_hist)
        inputList.append(input_hist)

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = shared_embedding_pos(input_pos)
        conList.append(embedding_pos)
        inputList.append(input_pos)

        input_day = Input(batch_shape=(batch_size, tweetLength,))
        input_hour = Input(batch_shape=(batch_size, tweetLength,))
        embedding_day = shared_embedding_day(input_day)
        embedding_hour = shared_embedding_hour(input_hour)
        conList.append(embedding_day)
        conList.append(embedding_hour)
        inputList.append(input_day)
        inputList.append(input_hour)

    embedding_comb = concatenate(conList)
    lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(embedding_comb)
    self_attention = SeqSelfAttention(attention_activation='sigmoid')(lstm)
    last_timestep = Lambda(lambda x: x[:, -1, :])(self_attention)
    output = Dense(labelNum, activation='softmax', name='output')(last_timestep)
    model = Model(inputs=inputList, outputs=output)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        tweetVector_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
        days_train = days_train[:-(len(days_train) % batch_size)]
        hours_train = hours_train[:-(len(hours_train) % batch_size)]
        posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        for i in range(histNum):
            histTweetVectors_train[i] = histTweetVectors_train[i][:-(len(histTweetVectors_train[i]) % batch_size)]
            histDayVectors_train[i] = histDayVectors_train[i][:-(len(histDayVectors_train[i]) % batch_size)]
            histHourVectors_train[i] = histHourVectors_train[i][:-(len(histHourVectors_train[i]) % batch_size)]
            histPOSVectors_train[i] = histPOSVectors_train[i][:-(len(histPOSVectors_train[i]) % batch_size)]
    if len(labels_val) % batch_size != 0:
        tweetVector_val = tweetVector_val[:-(len(tweetVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        days_val = days_val[:-(len(days_val) % batch_size)]
        hours_val = hours_val[:-(len(hours_val) % batch_size)]
        posVector_val = posVector_val[:-(len(posVector_val) % batch_size)]
        for i in range(histNum):
            histTweetVectors_val[i] = histTweetVectors_val[i][:-(len(histTweetVectors_val[i]) % batch_size)]
            histDayVectors_val[i] = histDayVectors_val[i][:-(len(histDayVectors_val[i]) % batch_size)]
            histHourVectors_val[i] = histHourVectors_val[i][:-(len(histHourVectors_val[i]) % batch_size)]
            histPOSVectors_val[i] = histPOSVectors_val[i][:-(len(histPOSVectors_val[i]) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    dataVector_train = [tweetVector_train, posVector_train, days_train, hours_train]
    dataVector_val = [tweetVector_val, posVector_val, days_val, hours_val]

    for i in range(histNum):
        dataVector_train += [histTweetVectors_train[i], histPOSVectors_train[i], histDayVectors_train[i], histHourVectors_train[i]]
        dataVector_val += [histTweetVectors_val[i], histPOSVectors_val[i], histDayVectors_val[i], histHourVectors_val[i]]

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size,
                                 class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size,
                                 verbose=verbose)

    accuracyHist = trainHistory.history['val_acc']
    lossHist = trainHistory.history['val_loss']

    tuneFile = open(resultName + '.tune', 'a')
    tuneFile.write('Hist Num: ' + str(histNum) + '\n')
    for index, loss in enumerate(lossHist):
        tuneFile.write(str(index + 1) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
    tuneFile.write('\n')
    tuneFile.close()

    scores = model.evaluate(dataVector_val, labelVector_val, batch_size=batch_size, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    predictions = model.predict(dataVector_val, batch_size=batch_size)
    sampleFile = open(resultName + '.sample', 'a')
    predLabels = []
    trueLabel_val = encoder.inverse_transform(labels_val)
    for index, pred in enumerate(predictions):
        predLabel = labelList[pred.tolist().index(max(pred))]
        sampleFile.write(ids_val[index] + '\t' + contents_val[index] + '\t' + trueLabel_val[index] + '\t' + predLabel + '\t' + places_val[index] + '\n')
        predLabels.append(predLabel)
    sampleFile.close()
    eval.addEval(scores[1], trueLabel_val, predLabels)

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


if __name__ == "__main__":
    modelName = 'long1.5'
    histName = 'long1.5'
    embModel = 'glove'
    #processContextTLSTM(modelName, 'none', embModel, char=False, epochs=20, dev=True)
    #processContextTLSTM(modelName, 'class', embModel, char=False, epochs=8, dev=True)

    #processContextPOSLSTM(modelName, 'none', embModel, char=False, epochs=20, dev=True)
    #processContextPOSLSTM(modelName, 'class', embModel, char=False, epochs=7, dev=True)

    #processContextPOSTLSTM(modelName, 'none', embModel, char=False, epochs=20, dev=True)
    #processContextPOSTLSTM(modelName, 'class', embModel, char=False, epochs=7, dev=False)
    processContextAttPOSTLSTM(modelName, 'class', embModel, char=False, epochs=5, dev=False)

    #processContextHistLSTM(modelName, histName, 'none', 'glove', char=False, histNum=5, epochs=20, dev=False)
    #processContextHistLSTM(modelName, histName, 'class', 'glove', char=False, histNum=5, epochs=10, dev=True)
    processContextAttHistLSTM(modelName, histName, 'class', 'glove', char=False, histNum=5, epochs=28, dev=False)

    #processContextHistPOSTLSTM('long1.5', 'none', 'glove', char=False, histNum=5, epochs=4, tune=False)
    #processContextHistPOSTLSTM('long1.5', 'long1.5', 'class', 'glove', char=False, histNum=5, epochs=12, dev=False)
    #processContextAttHistPOSTLSTM('long1.5', 'long1.5', 'class', 'glove', char=False, histNum=5, epochs=22, dev=False)

    #for num in [3]:
    #    processHistLSTM_period('long1.5', 'class', 'glove', periodNum=num, epochs=6, tune=False)