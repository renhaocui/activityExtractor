import json, re, sys, time, datetime
import numpy as np
from keras.layers import Dense, LSTM, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
from utilities import word2vecReader
from keras.layers.wrappers import TimeDistributed
from utilities import tokenizer, evaluation
from keras.layers import Layer
from keras import backend as K
from keras import initializers
from gensim.models import Word2Vec
from wordsegment import load, segment
load()
reload(sys)
sys.setdefaultencoding('utf8')

vocabSize = 10000
tweetLength = 25
posEmbLength = 25
embeddingVectorLength = 200
embeddingPOSVectorLength = 20
charLengthLimit = 20
batch_size = 100

dayMapper = {'Mon': 'DDMON', 'Tue': 'DDTUE', 'Wed': 'DDWED', 'Thu': 'DDTHU', 'Fri': 'DDFRI', 'Sat': 'DDSAT', 'Sun': 'DDSUN'}
POSMapper = {'N': 'N', 'O': 'O', '^': 'AA', 'S': 'S', 'Z': 'Z', 'L': 'L', 'M': 'M',
             'V': 'V', 'A': 'A', 'R': 'R', '@': 'BB', '#': 'CC', '~': 'DD', 'E': 'E', ',': 'EE', 'U': 'U',
             '!': 'FF', 'D': 'D', 'P': 'P', '&': 'GG', 'T': 'T', 'X': 'X', 'Y': 'Y', '$': 'HH', 'G': 'G'}
POSMapper2 = {'N': 'N', 'O': 'O', 'AA': '^', 'S': 'S', 'Z': 'Z', 'L': 'L', 'M': 'M',
             'V': 'V', 'A': 'A', 'R': 'R', 'BB': '@', 'CC': '#', 'DD': '~', 'E': 'E', 'EE': ',', 'U': 'U',
             'FF': '!', 'D': 'D', 'P': 'P', 'GG': '&', 'T': 'T', 'X': 'X', 'Y': 'Y', 'HH': '$', 'G': 'G'}

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        #self.W = self.init((input_shape[-1],1))
        self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*weights.dimshuffle(0,1,'x')
        return weighted_input.sum(axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[-1])


def hourMapper(hour):
    input = int(hour)
    if 0 <= input < 6:
        output = 'HHELMO'
    elif 6 <= input < 12:
        output = 'HHMORN'
    elif 12 <= input < 18:
        output = 'HHAFTE'
    else:
        output = 'HHNIGH'
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


def genTimeStr(input, length):
    output = ''
    for i in range(length):
        output += input + ' '
    return output.strip()


def copyPadding(inputList, sampleList):
    output = []
    for index, sampleVector in enumerate(sampleList):
        outputVector = []
        inputVector = inputList[index]
        for i, item in enumerate(sampleVector):
            if item == '0':
                outputVector.append(0)
            else:
                outputVector.append(inputVector[i])
        output.append(outputVector)
    return np.array(output)


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
                    posOutput += 'PPOOSSE' + ' '
            else:
                contentOutput += item[0] + ' '
                if mode == 'all':
                    posOutput += 'PPOOSS' + POSMapper[item[1]] + ' '
                else:
                    posOutput += 'PPOOSS' + POSMapper[item[1]] + ' '
        else:
            contentOutput += item[0] + ' '
            if mode == 'all':
                posOutput += 'PPOOSS' + POSMapper[item[1]] + ' '
            else:
                posOutput += 'PPOOSS' + POSMapper[item[1]] + ' '
        if len(contentOutput.split(' ')) != len(posOutput.split(' ')):
            print('error')
            print(contentOutput)
    return contentOutput.lower().strip().encode('utf-8'), posOutput.strip().encode('utf-8')


#order=1 -> pos, word
def mixPOS(inputList, mode, order=1):
    output = ''
    for item in inputList:
        if order != 1:
            output += item[0] + ' '
        if mode == 'all':
            output += 'PPOOSS'+POSMapper[item[1]] + ' '
        else:
            output += 'PPOOSS' + POSMapper[item[1]] + ' '
        if order == 1:
            output += item[0] + ' '
    return output.strip().encode('utf-8')


def loadData(modelName, char, embedding, hashtag, posMode='all'):
    activityList = []
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for line in activityListFile:
        if not line.startswith('NONE'):
            activityList.append(line.strip())
    activityListFile.close()
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()

    contents = []
    labels = []
    posList = []
    dayList = []
    hourList = []
    places = []
    idTagMapper = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if hashtag:
            POSFile = open('data/POS/' + place + '.pos', 'r')
        else:
            POSFile = open('data/POSnew/' + place + '.pos', 'r')
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
                content, pos = extractPOS(idTagMapper[id], mode=posMode)
                contents.append(content)
                posList.append(pos)
                dateTemp = data['created_at'].split()
                day = dayMapper[dateTemp[0]]
                hour = hourMapper(dateTemp[3].split(':')[0])
                dayList.append(day)
                hourList.append(hour)
                labels.append(activity)
                places.append(place)
        tweetFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    totalList = contents + posList
    tk.fit_on_texts(totalList)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    posSequences = tk.texts_to_sequences(posList)
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
            embVector = embeddings_index.get(word)
            if embVector is not None:
                embMatrix[i] = embVector
            '''
            if word in embeddings_index:
                embVector = embeddings_index[word]
                embMatrix[i] = embVector
            elif word[:6] == 'ppooss' and word[6:].upper() in POSMapper2:
                if POSMapper2[word[6:].upper()] in posEmbModel.wv:
                    embMatrix[i] = posEmbModel.wv[POSMapper2[word[6:].upper()]]
            '''
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

    return placeList, activityList, posList, dayList, hourList, labels, places, contents, idTagMapper, tweetVector, posVector, embMatrix, word_index


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



def processPOSLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, epochs=4):
    placeList, activityList, posList, dayList, hourList, labels, places, contents, idTagMapper, tweetVector, posVector, embMatrix, word_index = loadData(modelName,
                                                                                                                                                                       char,
                                                                                                                                                                       embedding,
                                                                                                                                                                       hashtag, posMode=posMode)
    labelNum = len(np.unique(activityList))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/H-POSLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()
    places = np.array(places)

    dataVector = []
    for index, tweet in enumerate(tweetVector):
        dataVector.append([posVector[index], tweet])
    dataVector = np.array(dataVector)

    # training
    print('training...')
    resultFile = open('result/H-POSLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dataVector, labels)):
        input = Input(batch_shape=(batch_size, 2, tweetLength,), name='input')
        if embedding in ['glove', 'word2vec']:
            embedding = TimeDistributed(Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True))(input)
        else:
            embedding = TimeDistributed(Embedding(vocabSize, embeddingVectorLength))(input)
        lower_lstm = TimeDistributed(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(embedding)

        higher_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(lower_lstm)
        output = Dense(labelNum, activation='softmax', name='output')(higher_lstm)
        model = Model(inputs=input, outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        data_train = dataVector[train_index]
        labels_train = labels[train_index]

        data_test = dataVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]

        if len(labels_train) % batch_size != 0:
            data_train = data_train[:-(len(data_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            data_test = data_test[:-(len(data_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate(data_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(data_test, batch_size=batch_size)
        sampleFile = open('result/H-POSLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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

    sumConMatrix = np.divide(sumConMatrix, 5)
    confusionFile = open('result/H-POSLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processTLSTM(modelName, balancedWeight='None', embedding='None', char=False, hashtag=True, epochs=4):
    activityList = []
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for line in activityListFile:
        if not line.startswith('NONE'):
            activityList.append(line.strip())
    activityListFile.close()
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()
    labelNum = len(np.unique(activityList))

    contents = []
    labels = []
    labelCount = {}
    dayList = []
    hourList = []
    idTagMapper = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if hashtag:
            POSFile = open('data/POS/' + place + '.pos', 'r')
        else:
            POSFile = open('data/POSnew/' + place + '.pos', 'r')
        for line in POSFile:
            data = json.loads(line.strip())
            idTagMapper[data['id']] = data['tag']
        POSFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                content, pos = extractPOS(idTagMapper[id], 'all')
                contents.append(content)
                dateTemp = data['created_at'].split()
                day = dayMapper[dateTemp[0]]
                dayList.append(genTimeStr(day, tweetLength))
                hour = hourMapper(dateTemp[3].split(':')[0])
                hourList.append(genTimeStr(hour, tweetLength))
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
    labelFile = open('result/H-TLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    totalList = contents + dayList + hourList
    tk.fit_on_texts(totalList)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    daySequences = tk.texts_to_sequences(dayList)
    dayVector = copyPadding(daySequences, tweetVector)
    #dayVector = sequence.pad_sequences(daySequences, maxlen=tweetLength, truncating='post', padding='post')

    hourSequences = tk.texts_to_sequences(hourList)
    hourVector = copyPadding(hourSequences, tweetVector)
    #hourVector = sequence.pad_sequences(hourSequences, maxlen=tweetLength, truncating='post', padding='post')

    dataVector = []
    for index, tweet in enumerate(tweetVector):
        dataVector.append([dayVector[index], hourVector[index], tweet])
    dataVector = np.array(dataVector)

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
    resultFile = open('result/H-TLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dataVector, labels)):
        input = Input(batch_shape=(batch_size, 3, tweetLength,), name='input')
        if embedding in ['glove', 'word2vec']:
            embedding = TimeDistributed(Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True))(input)
        else:
            embedding = TimeDistributed(Embedding(vocabSize, embeddingVectorLength))(input)
        lower_lstm = TimeDistributed(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(embedding)

        higher_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(lower_lstm)
        output = Dense(labelNum, activation='softmax', name='output')(higher_lstm)
        model = Model(inputs=input, outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        data_train = dataVector[train_index]
        labels_train = labels[train_index]

        data_test = dataVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            data_train = data_train[:-(len(data_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            data_test = data_test[:-(len(data_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate(data_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(data_test, batch_size=batch_size)
        sampleFile = open('result/H-TLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/H-TLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHistLSTM_time(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, epochs=4):
    activityList = []
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for line in activityListFile:
        if not line.startswith('NONE'):
            activityList.append(line.strip())
    activityListFile.close()
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()
    labelNum = len(np.unique(activityList))

    contents = []
    labels = []
    histContents = []
    idTagMapper = {}
    print('Loading...')
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if hashtag:
            POSFile = open('data/POS/' + place + '.pos', 'r')
        else:
            POSFile = open('data/POSnew/' + place + '.pos', 'r')
        for line in POSFile:
            data = json.loads(line.strip())
            idTagMapper[data['id']] = data['tag']
        POSFile.close()

        histMap = {}
        histFile = open('data/POIHistClean/' + place + '.json', 'r')
        for line in histFile:
            histData = json.loads(line.strip())
            id = histData['max_id']
            for tweet in reversed(histData['statuses']):
                text = tweet['text']
                dateTemp = tweet['created_at'].split()
                day = dayMapper[dateTemp[0]]
                hour = hourMapper(dateTemp[3].split(':')[0])
                if tweet['id'] != id:
                    histMap[id] = {hour: text}
        histFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                # content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                content, pos = extractPOS(idTagMapper[id], posMode)
                dateTemp = data['created_at'].split()
                day = dayMapper[dateTemp[0]]
                hour = hourMapper(dateTemp[3].split(':')[0])
                if id in histMap:
                    if hour in histMap[id]:
                        histContent = cleanContent(histMap[id][hour], hashtag=False, breakEmoji=True)
                        histContents.append(histContent)
                        contents.append(content)
                        labels.append(activity)
        tweetFile.close()

    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/Hist_TimeLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    totalList = contents + histContents
    tk.fit_on_texts(totalList)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    histSequences = tk.texts_to_sequences(histContents)
    histVector = sequence.pad_sequences(histSequences, maxlen=tweetLength, truncating='post', padding='post')

    dataVector = []
    for index, tweet in enumerate(tweetVector):
        dataVector.append([histVector[index], tweet])
    dataVector = np.array(dataVector)

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
    resultFile = open('result/Hist_TimeLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dataVector, labels)):
        input = Input(batch_shape=(batch_size, 2, tweetLength,), name='input')
        if embedding in ['glove', 'word2vec']:
            embedding = TimeDistributed(Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True))(input)
        else:
            embedding = TimeDistributed(Embedding(vocabSize, embeddingVectorLength))(input)
        lower_lstm = TimeDistributed(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(embedding)

        higher_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(lower_lstm)
        output = Dense(labelNum, activation='softmax', name='output')(higher_lstm)
        model = Model(inputs=input, outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        data_train = dataVector[train_index]
        labels_train = labels[train_index]

        data_test = dataVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            data_train = data_train[:-(len(data_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            data_test = data_test[:-(len(data_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate(data_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(data_test, batch_size=batch_size)
        sampleFile = open('result/Hist_TimeLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/Hist_TimeLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHistLSTM(modelName, histName, balancedWeight='None', embedding='None', char=False, histNum=1, epochs=4, dev=False):
    resultName = 'result/H-HistLSTM_' + modelName + '_' + balancedWeight
    ids_train, ids_val, labels_train, labels_val, places_train, places_val, contents_train, contents_val, days_train, days_val, hours_train, hours_val, \
    tweetVector_train, tweetVector_val, histTweetVectors_train, histTweetVectors_val, histDayVectors_train, histDayVectors_val, histHourVectors_train, histHourVectors_val, embMatrix, word_index = loadHistData(modelName, histName, char, embedding, histNum=histNum, pos=False, dev=dev)

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

    dataVector_train = []
    for index, tweet in enumerate(tweetVector_train):
        temp = []
        for vector in reversed(histTweetVectors_train):
            temp.append(vector[index])
        temp.append(tweet)
        dataVector_train.append(temp)
    dataVector_train = np.array(dataVector_train)

    dataVector_val = []
    for index, tweet in enumerate(tweetVector_val):
        temp = []
        for vector in reversed(histTweetVectors_val):
            temp.append(vector[index])
        temp.append(tweet)
        dataVector_val.append(temp)
    dataVector_val = np.array(dataVector_val)

    # training
    print('training...')
    if dev:
        verbose = 2
    else:
        verbose = 0
    eval = evaluation.evalMetrics(labelNum)

    input = Input(batch_shape=(batch_size, histNum+1, tweetLength,), name='input')
    if embedding in ['glove', 'word2vec']:
        embedding = TimeDistributed(Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True), name='embedding')(input)
    else:
        embedding = TimeDistributed(Embedding(vocabSize, embeddingVectorLength))(input)
    lower_lstm = TimeDistributed(LSTM(200, dropout=0.2, recurrent_dropout=0.2), name='lower_lstm')(embedding)

    higher_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='higher_lstm')(lower_lstm)
    output = Dense(labelNum, activation='softmax', name='output')(higher_lstm)
    model = Model(inputs=input, outputs=output)
    #print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if len(labels_train) % batch_size != 0:
        dataVector_train = dataVector_train[:-(len(dataVector_train) % batch_size)]
        labels_train = labels_train[:-(len(labels_train) % batch_size)]
    if len(labels_val) % batch_size != 0:
        dataVector_val = dataVector_val[:-(len(dataVector_val) % batch_size)]
        labels_val = labels_val[:-(len(labels_val) % batch_size)]
        places_val = places_val[:-(len(places_val) % batch_size)]
        ids_val = ids_val[:-(len(ids_val) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)
    labelVector_val = np_utils.to_categorical(labels_val)

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        trainHistory = model.fit(dataVector_train, labelVector_train, epochs=epochs, validation_data=(dataVector_val, labelVector_val), batch_size=batch_size, verbose=verbose)

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
    confusionFile = open(resultName + '.confMatrix', 'a')
    resultFile = open(resultName + '.result', 'a')
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
    resultName = 'result/H-HistLSTM_period_' + str(periodNum) + '_' + modelName + '_' + balancedWeight
    emptyVector = []
    for i in range(tweetLength):
        emptyVector.append(0)
    maxHistNum = 51
    minHistNum = 1
    histData = {}
    histFile = open('data/consolidateHistData_' + modelName + '_max.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        histData[int(data.keys()[0])] = data.values()[0]
    histFile.close()

    totalContents = []
    contents = []
    labels = []
    places = []
    ids = []
    trainData = []
    dataFile = open('data/consolidateData_' + modelName + '_CreatedAt.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        if data['id'] in histData:
            histTweets = histData[data['id']]
            if len(histTweets) >= minHistNum:
                tempData = []
                totalContents.append(data['content'].encode('utf-8'))
                contents.append(data['content'].encode('utf-8'))
                labels.append(data['label'])
                places.append(data['place'])
                ids.append(str(data['id']))
                timeTemp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
                createdTimestamp = time.mktime(datetime.datetime.strptime(timeTemp, '%Y-%m-%d %H:%M:%S').timetuple())
                for histTweet in reversed(histTweets):
                    timeTemp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(histTweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                    histCreatedTimestamp = time.mktime(datetime.datetime.strptime(timeTemp, '%Y-%m-%d %H:%M:%S').timetuple())
                    if (createdTimestamp - histCreatedTimestamp) < periodNum * 8 * 3600:
                        totalContents.append(histTweet['content'].encode('utf-8'))
                        tempData.append(histTweet['content'].encode('utf-8'))
                tempData.append(data['content'].encode('utf-8'))
                trainData.append(tempData)
    print len(trainData)
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

    dataVector = []
    for tempData in trainData:
        tweetSequences = tk.texts_to_sequences(tempData)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')
        #print len(tweetVector)
        if len(tweetVector) < maxHistNum:
            for i in range(maxHistNum-len(tweetVector)):
                tweetVector = np.append(tweetVector, [emptyVector], axis=0)
        dataVector.append(tweetVector)
    dataVector = np.array(dataVector)
    #print len(dataVector[0])
    print dataVector.shape

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
    if tune:
        verbose = 2
    else:
        verbose = 0
    eval = evaluation.evalMetrics(labelNum)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dataVector, labels)):
        input = Input(batch_shape=(batch_size, None, tweetLength,), name='input')
        if embedding in ['glove', 'word2vec']:
            embedding = TimeDistributed(Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True), name='embedding')(input)
        else:
            embedding = TimeDistributed(Embedding(vocabSize, embeddingVectorLength))(input)
        lower_lstm = TimeDistributed(LSTM(200, dropout=0.2, recurrent_dropout=0.2), name='lower_lstm')(embedding)

        higher_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='higher_lstm')(lower_lstm)
        output = Dense(labelNum, activation='softmax', name='output')(higher_lstm)
        model = Model(inputs=input, outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        data_train = dataVector[train_index]
        labels_train = labels[train_index]

        data_test = dataVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]
        ids_test = ids[test_index]

        if len(labels_train) % batch_size != 0:
            data_train = data_train[:-(len(data_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            data_test = data_test[:-(len(data_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            ids_test = ids_test[:-(len(ids_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            trainHistory = model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            trainHistory = model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, class_weight=classWeight, verbose=verbose)
        else:
            trainHistory = model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, verbose=verbose)

        accuracyHist = trainHistory.history['val_acc']
        lossHist = trainHistory.history['val_loss']

        tuneFile = open(resultName + '.tune', 'a')
        tuneFile.write('Period Num: ' + str(periodNum) + '\n')
        for index, loss in enumerate(lossHist):
            tuneFile.write(str(index + 1) + '\t' + str(loss) + '\t' + str(accuracyHist[index]) + '\n')
        tuneFile.write('\n')
        tuneFile.close()

        scores = model.evaluate(data_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        predictions = model.predict(data_test, batch_size=batch_size)
        sampleFile = open(resultName + '.sample', 'a')
        predLabels = []
        for index, pred in enumerate(predictions):
            predLabel = labelList[pred.tolist().index(max(pred))]
            if not tune:
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
        confusionFile = open(resultName + '.confMatrix', 'a')
        resultFile = open(resultName + '.result', 'a')
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


def processMIXLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', epochs=4):
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
                    content = mixPOS(idTagMapper[id], posMode, 1)
                    #content, pos = extractPOS(idTagMapper[id], posMode)
                    contents.append(content)
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
    labelFile = open('result/MIXLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

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
        modelFile = '../tweetEmbeddingData/pos.word2vec'
        posEmbModel = Word2Vec.load(modelFile)

        word_index = tk.word_index
        embMatrix = np.zeros((len(word_index) + 1, 200))
        for word, i in word_index.items():
            if word in embeddings_index:
                embVector = embeddings_index[word]
                embMatrix[i] = embVector
            elif word[:6] == 'ppooss' and word[6:].upper() in POSMapper2:
                if POSMapper2[word[6:].upper()] in posEmbModel.wv:
                    embMatrix[i] = posEmbModel.wv[POSMapper2[word[6:].upper()]]

    # training
    print('training...')
    resultFile = open('result/MIXLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]

        inputs = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
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
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweet_test, labelVector_test), sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweet_test, labelVector_test), class_weight=classWeight, verbose=0)
        else:
            model.fit(tweet_train, labelVector_train, epochs=epochs, batch_size=batch_size, validation_data=(tweet_test, labelVector_test), verbose=2)

        scores = model.evaluate(tweet_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        score += scores[1] * 100
        predictions = model.predict(tweet_test, batch_size=batch_size)
        sampleFile = open('result/LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/MIXLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processCHLTM(modelName, balancedWeight='None', epochs=4):
    vocabSize = 70
    wordLength = 20
    embeddingWordLength = 200
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
                    contents.append(removeLinks(content))
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
    labelFile = open('result/CHLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    tk = Tokenizer(num_words=vocabSize, char_level=True, filters='')
    tk.fit_on_texts(contents)
    dataVector = []
    emptyWord = np.zeros(wordLength, dtype='int')
    for index, content in enumerate(contents):
        data = tokenizer.simpleTokenize(content)
        tweetSequences = tk.texts_to_sequences(data)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=wordLength, truncating='post', padding='post')
        if len(tweetVector) >= tweetLength:
            tweetVector = tweetVector[:tweetLength]
        else:
            for i in range(tweetLength-len(tweetVector)):
                tweetVector = np.vstack([tweetVector, emptyWord])
        dataVector.append(tweetVector)

    dataVector = np.array(dataVector)
    print dataVector.shape

    # training
    print('training...')
    resultFile = open('result/CHLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dataVector, labels)):
        input = Input(batch_shape=(batch_size, tweetLength, wordLength,), name='input')
        embedding = TimeDistributed(Embedding(vocabSize, embeddingWordLength))(input)
        lower_lstm = TimeDistributed(LSTM(200, dropout=0.2, recurrent_dropout=0.2))(embedding)
        
        higher_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(lower_lstm)
        output = Dense(labelNum, activation='softmax', name='output')(higher_lstm)
        model = Model(inputs=input, outputs=output)
        #print model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        data_train = dataVector[train_index]
        labels_train = labels[train_index]

        data_test = dataVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            data_train = data_train[:-(len(data_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            data_test = data_test[:-(len(data_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(data_train, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(data_train, labelVector_train, epochs=epochs, validation_data=(data_test, labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate(data_test, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(data_test, batch_size=batch_size)
        sampleFile = open('result/CHLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/CHLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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
    print(precision * 100 / 5)
    print(f1 * 100 / 5)
    resultFile.close()



if __name__ == '__main__':
    modelName = 'long1.5'
    histName = 'long1.5'
    embModel = 'glove'
    #processPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=True, epochs=9)
    #processPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=11)
    #processPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='map', epochs=10)

    #processTLSTM('long1.5', 'none', 'glove', char=False, hashtag=True, epochs=6)
    #processTLSTM('long1.5', 'none', 'glove', char=False, hashtag=False, epochs=6)

    #processHistLSTM_time('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=16)

    #processHistLSTM(modelName, histName, 'none', embModel, char=False, histNum=5, epochs=15, dev=False)
    processHistLSTM(modelName, histName, 'class', embModel, char=False, histNum=5, epochs=15, dev=True)

    #processMIXLSTM('long1.5', 'none', 'glove', char=False, posMode='all', epochs=7)
    #processMIXLSTM('long1.5', 'class', 'none', char=False, posMode='all', epochs=3)

    #processMIXLSTM('long1.5', 'none', 'none', char=False, posMode='map', epochs=3)
    #processMIXLSTM('long1.5', 'class', 'none', char=False, posMode='map', epochs=3)

    #processCHLTM('long1.5', 'none', epochs=200)


    #for num in [9]:
    #    processHistLSTM_period('long1.5', 'none', 'glove', periodNum=num, epochs=30, tune=True)
    #    processHistLSTM_period('long1.5', 'class', 'glove', periodNum=num, epochs=30, tune=True)