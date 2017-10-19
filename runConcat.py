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
                content, pos = extractPOS(idTagMapper[id])
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
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    if posMode == 'all':
        posVocabSize = 25
    else:
        posVocabSize = 11

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

    return placeList, activityList, posList, dayList, hourList, labels, places, contents, idTagMapper, tweetVector, posVector, posVocabSize, embMatrix, word_index


def processTLSTM(modelName, balancedWeight='None', embedding='None', char=False, hashtag=True, epochs=4):
    placeList, activityList, posList, dayList, hourList, labels, places, contents, idTagMapper, tweetVector, posVector, posVocabSize, embMatrix, word_index = loadData(modelName, char, embedding, hashtag)

    labelNum = len(np.unique(activityList))
    dayVector = to_categorical(dayList, num_classes=7)
    hourVector = to_categorical(hourList, num_classes=4)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/TLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    print('training...')
    resultFile = open('result/TLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
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

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit([tweet_train, hour_train, day_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, hour_train, day_train], labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit([tweet_train, hour_train, day_train], labelVector_train, validation_data=([tweet_test, hour_test, day_test], labelVector_test), epochs=epochs, batch_size=batch_size,
                      verbose=2)

        scores = model.evaluate([tweet_test, hour_test, day_test], labelVector_test, batch_size=batch_size, verbose=2)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, hour_test, day_test], batch_size=batch_size)
        sampleFile = open('result/TLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/TLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processCLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, epochs=4):
    placeList, activityList, posList, dayList, hourList, labels, places, contents, idTagMapper, tweetVector, posVector, posVocabSize, embMatrix, word_index = loadData(modelName, char, embedding,
                                                                                                                                                         hashtag, posMode=posMode)
    labelNum = len(np.unique(activityList))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/CLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    print('training...')
    resultFile = open('result/CLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
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
        sampleFile = open('result/CLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/CLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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

def processHistLSTM(modelName, balancedWeight='None', embedding='None', char=False, hashtag=True, histNum=1, epochs=7):
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
    places = []
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
                    tempText.append(histData['statuses'][i + 1]['text'])
                histMap[id] = tempText
        histFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                content, pos = extractPOS(idTagMapper[id], 'all')
                if id in histMap:
                    for i in range(histNum):
                        histContent = cleanContent(histMap[id][i], hashtag=hashtag, breakEmoji=True)
                        histContents[i].append(histContent)
                    contents.append(content)
                    labels.append(activity)
                    places.append(place)
        tweetFile.close()

    places = np.array(places)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-HistLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
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
    resultFile = open('result/C-HistLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
        if embedding in ['glove', 'word2vec']:
            shared_embedding = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
            embedding_tweet = shared_embedding(input_tweet)
        else:
            shared_embedding = Embedding(vocabSize, embeddingVectorLength)
            embedding_tweet = shared_embedding(input_tweet)
        tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_tweet)

        conList = [tweet_lstm]
        inputList = [input_tweet]
        for i in range(histNum):
            input_hist = Input(batch_shape=(batch_size, posEmbLength,))
            embedding_hist = shared_embedding(input_hist)
            hist_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_hist)
            conList.append(hist_lstm)
            inputList.append(input_hist)

        comb = concatenate(conList)
        output = Dense(labelNum, activation='softmax', name='output')(comb)
        model = Model(inputs=inputList, outputs=output)
        #print(model.summary())
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]
        histVector_train = []
        histVector_test = []
        for i in range(histNum):
            histVector_train.append(histVectors[i][train_index])
            histVector_test.append(histVectors[i][test_index])

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            for i in range(histNum):
                histVector_train[i] = histVector_train[i][:-(len(histVector_train[i]) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            for i in range(histNum):
                histVector_test[i] = histVector_test[i][:-(len(histVector_test[i]) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        trainList = [tweet_train]
        testList = [tweet_test]
        for i in range(histNum):
            trainList.append(histVector_train[i])
            testList.append(histVector_test[i])

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=2)

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-HistLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-HistLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHistLSTM_contextT(modelName, balancedWeight='None', embedding='None', char=False, hashtag=True, histNum=1, epochs=7):
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
    histDayVector = {}
    histHourVector = {}
    idTagMapper = {}
    dayVector = []
    hourVector = []
    places = []
    for i in range(histNum):
        histContents[i] = []
        histDayVector[i] = []
        histHourVector[i] = []
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
                tempData = []
                for i in range(histNum):
                    tweet = histData['statuses'][i + 1]
                    text = tweet['text']
                    dateTemp = tweet['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    tempData.append([text, day, hour])
                histMap[id] = tempData
        histFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                content, pos = extractPOS(idTagMapper[id], 'all')
                if id in histMap:
                    for i in range(histNum):
                        histContents[i].append(cleanContent(histMap[id][i][0], hashtag=hashtag, breakEmoji=True))
                        histDayVector[i].append(np.full((tweetLength), histMap[id][i][1], dtype='int'))
                        histHourVector[i].append(np.full((tweetLength), histMap[id][i][2], dtype='int'))
                    contents.append(content)
                    dateTemp = data['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    dayVector.append(np.full((tweetLength), day, dtype='int'))
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    hourVector.append(np.full((tweetLength), hour, dtype='int'))
                    labels.append(activity)
                    places.append(place)
        tweetFile.close()

    for i in range(histNum):
        histDayVector[i] = np.array(histDayVector[i])
        histHourVector[i] = np.array(histHourVector[i])

    dayVector = np.array(dayVector)
    hourVector = np.array(hourVector)
    places = np.array(places)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
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

    histTweetVectors = []
    for i in range(histNum):
        histSequence = tk.texts_to_sequences(histContents[i])
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors.append(tempVector)

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
    resultFile = open('result/C-Hist-Context-T-LSTM_.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
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
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=2)

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-Hist-Context-T-LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processHistLSTM_contextPOS(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, histNum=1, epochs=7, tune=False):
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
    posList = []
    histContents = {}
    histPOSList = {}
    idTagMapper = {}
    places = []
    for i in range(histNum):
        histContents[i] = []
        histPOSList[i] = []
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

        POShistFile = open('data/POShistClean/'+place+'.pos', 'r')
        for line in POShistFile:
            data = json.loads(line.strip())
            idTagMapper[int(data.keys()[0])] = data.values()[0]
        POShistFile.close()

        histMap = {}
        histFile = open('data/POIHistClean/' + place + '.json', 'r')
        for line in histFile:
            histData = json.loads(line.strip())
            id = histData['max_id']
            if len(histData['statuses']) > histNum:
                tempData = []
                for i in range(histNum):
                    tweet = histData['statuses'][i + 1]
                    text = tweet['text']
                    histID = tweet['id']
                    dateTemp = tweet['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    tempData.append([text, day, hour, histID])
                histMap[id] = tempData
        histFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                content, pos = extractPOS(idTagMapper[id], 'all')
                if id in histMap:
                    suffHist = True
                    for i in range(histNum):
                        if histMap[id][i][3] not in idTagMapper:
                            suffHist = False
                            break
                    if suffHist:
                        for i in range(histNum):
                            histContent, histPOS = extractPOS(idTagMapper[histMap[id][i][3]], 'all')
                            histContents[i].append(histContent)
                            histPOSList[i].append(histPOS)
                        contents.append(content)
                        posList.append(pos)
                        labels.append(activity)
                        places.append(place)
        tweetFile.close()

    places = np.array(places)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
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

    histTweetVectors = []
    for i in range(histNum):
        histSequence = tk.texts_to_sequences(histContents[i])
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors.append(tempVector)

    if posMode == 'all':
        posVocabSize = 25
    else:
        posVocabSize = 11

    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    totalPOSList = posList[:]
    for i in range(histNum):
        totalPOSList += histPOSList[i]
    tkPOS.fit_on_texts(totalPOSList)
    posSequences = tkPOS.texts_to_sequences(posList)
    posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

    histPOSVectors = []
    for i in range(histNum):
        histPOSSequences = tkPOS.texts_to_sequences(histPOSList[i])
        histPOSVector = sequence.pad_sequences(histPOSSequences, maxlen=tweetLength, truncating='post', padding='post')
        histPOSVectors.append(histPOSVector)

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
    resultFile = open('result/C-Hist-Context-POS-LSTM_.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
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
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=2)

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-Hist-Context-POS-LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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



def processHistLSTM_contextPOST(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, histNum=1, epochs=7):
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
    posList = []
    histContents = {}
    histDayVector = {}
    histHourVector = {}
    histPOSList = {}
    idTagMapper = {}
    dayVector = []
    hourVector = []
    places = []
    for i in range(histNum):
        histContents[i] = []
        histDayVector[i] = []
        histHourVector[i] = []
        histPOSList[i] = []
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

        POShistFile = open('data/POShistClean/'+place+'.pos', 'r')
        for line in POShistFile:
            data = json.loads(line.strip())
            idTagMapper[int(data.keys()[0])] = data.values()[0]
        POShistFile.close()

        histMap = {}
        histFile = open('data/POIHistClean/' + place + '.json', 'r')
        for line in histFile:
            histData = json.loads(line.strip())
            id = histData['max_id']
            if len(histData['statuses']) > histNum:
                tempData = []
                for i in range(histNum):
                    tweet = histData['statuses'][i + 1]
                    text = tweet['text']
                    histID = tweet['id']
                    dateTemp = tweet['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    tempData.append([text, day, hour, histID])
                histMap[id] = tempData
        histFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > charLengthLimit:
                id = data['id']
                content, pos = extractPOS(idTagMapper[id], 'all')
                if id in histMap:
                    suffHist = True
                    for i in range(histNum):
                        if histMap[id][i][3] not in idTagMapper:
                            suffHist = False
                            break
                    if suffHist:
                        for i in range(histNum):
                            histContent, histPOS = extractPOS(idTagMapper[histMap[id][i][3]], 'all')
                            histContents[i].append(histContent)
                            histDayVector[i].append(np.full((tweetLength), histMap[id][i][1], dtype='int'))
                            histHourVector[i].append(np.full((tweetLength), histMap[id][i][2], dtype='int'))
                            histPOSList[i].append(histPOS)
                        contents.append(content)
                        dateTemp = data['created_at'].split()
                        day = dayMapper[dateTemp[0]]
                        dayVector.append(np.full((tweetLength), day, dtype='int'))
                        hour = hourMapper(dateTemp[3].split(':')[0])
                        hourVector.append(np.full((tweetLength), hour, dtype='int'))
                        posList.append(pos)
                        labels.append(activity)
                        places.append(place)
        tweetFile.close()

    for i in range(histNum):
        histDayVector[i] = np.array(histDayVector[i])
        histHourVector[i] = np.array(histHourVector[i])
    dayVector = np.array(dayVector)
    hourVector = np.array(hourVector)
    places = np.array(places)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
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

    histTweetVectors = []
    for i in range(histNum):
        histSequence = tk.texts_to_sequences(histContents[i])
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors.append(tempVector)

    if posMode == 'all':
        posVocabSize = 25
    else:
        posVocabSize = 11

    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    totalPOSList = posList[:]
    for i in range(histNum):
        totalPOSList += histPOSList[i]
    tkPOS.fit_on_texts(totalPOSList)
    posSequences = tkPOS.texts_to_sequences(posList)
    posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

    histPOSVectors = []
    for i in range(histNum):
        histPOSSequences = tkPOS.texts_to_sequences(histPOSList[i])
        histPOSVector = sequence.pad_sequences(histPOSSequences, maxlen=tweetLength, truncating='post', padding='post')
        histPOSVectors.append(histPOSVector)

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
    resultFile = open('result/C-Hist-Context-POST-LSTM_.' + modelName + '_' + balancedWeight + '.result', 'a')
    score = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    sumConMatrix = np.zeros([labelNum, labelNum])
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
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size,
                      verbose=2)

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


if __name__ == "__main__":
    #processTLSTM('long1.5', 'none', 'glove', char=False, hashtag=True, epochs=6)
    #processTLSTM('long1.5', 'none', 'glove', char=False, hashtag=False, epochs=6)

    #processCLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=True, epochs=8)
    #processCLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=6)
    #processCLSTM('long1.5', 'none', 'glove', char=False, posMode='map', posEmbedding=False, epochs=9)

    #processMLSTM('long1.5', 'none', 'none', char=False, posMode='all', epochs=4)
    #processMLSTM('long1.5', 'class', 'none', char=False, posMode='all', epochs=4)

    #processMLSTM('long1.5', 'none', 'none', char=False, posMode='map', epochs=4)
    #processMLSTM('long1.5', 'class', 'none', char=False, posMode='map', epochs=4)

    #processMLSTM2('long1.5', 'none', 'none', char=False, posMode='all', epochs=10)

    #processHistLSTM('long1.5', 'none', 'glove', char=False, hashtag=False, histNum=2, epochs=8)
    #processHistLSTM_contextT('long1.5', 'none', 'glove', char=False, hashtag=False, histNum=3, epochs=20)
    #processHistLSTM_contextPOST('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, histNum=1, epochs=6)

    processHistLSTM_contextPOS('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, histNum=1, epochs=7, tune=False)
    processHistLSTM_contextPOS('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, histNum=4, epochs=13, tune=False)
