import json, re, sys, evaluation
import numpy as np
from keras.layers import Dense, LSTM, Input
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from utilities import word2vecReader
from utilities import tokenizer
from keras.layers import Layer
from keras import backend as K
from keras import initializers
from wordsegment import load, segment
load()
reload(sys)
sys.setdefaultencoding('utf8')

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


def processSeqConcatLSTM(modelName, balancedWeight='None', embedding='None', histNum=1, epochs=4, tune=False):
    print('Loading...')
    resultName = 'result/SeqConcat-HistLSTM_' + modelName + '_' + balancedWeight
    histData = {}
    histFile = open('data/consolidateHistData_' + modelName + '.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        histData[int(data.keys()[0])] = data.values()[0]

    totalContents = []
    labels = []
    places = []
    ids = []
    contents = []
    histContents = []
    dataFile = open('data/consolidateData_' + modelName + '.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        if data['id'] in histData:
            histTweets = histData[data['id']]
            if len(histTweets) >= histNum:
                tempHist = []
                contents.append(data['content'].encode('utf-8'))
                totalContents.append(data['content'].encode('utf-8'))
                labels.append(data['label'])
                places.append(data['place'])
                ids.append(str(data['id']))
                for i in range(histNum):
                    totalContents.append(histTweets[i]['content'].encode('utf-8'))
                    tempHist.append(histTweets[histNum-1-i]['content'].encode('utf-8'))
                histContents.append(tempHist)

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

    histVectors = []
    for tempHist in histContents:
        histSequence = tk.texts_to_sequences(tempHist)
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histVectors.append(tempVector)

    dataVector = []
    for index, tweet in enumerate(tweetVector):
        for i, histTweet in enumerate(histVectors[index]):
            if i == 0:
                tempSeq = histTweet
            else:
                tempSeq = np.append(tempSeq, histTweet)
        tempSeq = np.append(tempSeq, tweet)
        dataVector.append(tempSeq)
    dataVector = np.array(dataVector)

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
        input = Input(batch_shape=(batch_size, tweetLength*(histNum+1),), name='input')
        if embedding in ['glove', 'word2vec']:
            embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True, name='embedding')(input)
        else:
            embedding = Embedding(vocabSize, embeddingVectorLength)(input)
        lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, name='lstm')(embedding)
        output = Dense(labelNum, activation='softmax', name='output')(lstm)
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
        tuneFile.write('Hist Num: ' + str(histNum) + '\n')
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


if __name__ == '__main__':
    processSeqConcatLSTM('long1.5', 'none', 'glove', histNum=4, epochs=9, tune=False)
    processSeqConcatLSTM('long1.5', 'class', 'glove', histNum=5, epochs=11, tune=False)

