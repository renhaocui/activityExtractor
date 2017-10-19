import json, re
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
from utilities import tokenizer
from keras.layers import Layer
from keras import backend as K
from keras import initializers
from gensim.models import Word2Vec
from wordsegment import load, segment
load()

vocabSize = 10020
tweetLength = 25
posEmbLength = 25
embeddingVectorLength = 200
embeddingPOSVectorLength = 200
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


def processHistLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, histNum=1, epochs=4):
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
    histContents = {}
    idTagMapper = {}
    for i in range(histNum):
        histContents[i] = []
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
            if len(histData['statuses']) > histNum:
                tempText = []
                for i in range(histNum):
                    tempText.append(histData['statuses'][i+1]['text'])
                histMap[id] = tempText
        histFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                # content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                content, pos = extractPOS(idTagMapper[id], posMode)
                #dateTemp = data['created_at'].split()
                #day = dayMapper[dateTemp[0]]
                #hour = hourMapper(dateTemp[3].split(':')[0])
                if id in histMap:
                    for i in range(histNum):
                        histContent = cleanContent(histMap[id][i], hashtag=False, breakEmoji=True)
                        histContents[i].append(histContent)
                    contents.append(content)
                    labels.append(activity)
        tweetFile.close()

    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/H-HistLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

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

    histVectors = []
    for i in range(histNum):
        histSequence = tk.texts_to_sequences(histContents[i])
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histVectors.append(tempVector)
    dataVector = []
    for index, tweet in enumerate(tweetVector):
        temp = []
        for vector in reversed(histVectors):
            temp.append(vector[index])
        temp.append(tweet)
        dataVector.append(temp)
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
    resultFile = open('result/H-HistLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(dataVector, labels)):
        input = Input(batch_shape=(batch_size, histNum+1, tweetLength,), name='input')
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
        sampleFile = open('result/H-HistLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/H-HistLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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
    #processPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=True, epochs=9)
    ##processPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=11)
    #processPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='map', epochs=10)

    #processTLSTM('long1.5', 'none', 'glove', char=False, hashtag=True, epochs=6)
    ##processTLSTM('long1.5', 'none', 'glove', char=False, hashtag=False, epochs=6)

    processHistLSTM_time('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=16)
    #processHistLSTM('long1.5', 'none', 'glove', char=False, hashtag=False, histNum=3, epochs=5)

    #processMIXLSTM('long1.5', 'none', 'glove', char=False, posMode='all', epochs=7)
    #processMIXLSTM('long1.5', 'class', 'none', char=False, posMode='all', epochs=3)

    #processMIXLSTM('long1.5', 'none', 'none', char=False, posMode='map', epochs=3)
    #processMIXLSTM('long1.5', 'class', 'none', char=False, posMode='map', epochs=3)

    #processCHLTM('long1.5', 'none', epochs=200)
