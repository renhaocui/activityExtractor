import json, re
import numpy as np
from keras.layers import Dense, LSTM, Input, concatenate, Lambda, multiply, Flatten
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
from wordsegment import load, segment
from utilities import tokenizer

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


def processContextPOSLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, epochs=7):
    placeList, activityList, posList, dayList, hourList, labels, places, contents, idTagMapper, tweetVector, posVector, posVocabSize, embMatrix, word_index = loadData(modelName,
                                                                                                                                                                       char,
                                                                                                                                                                       embedding,
                                                                                                                                                                       hashtag,
                                                                                                                                                                       posMode=posMode)
    labelNum = len(np.unique(activityList))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/ContextPOSLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()
    places = np.array(places)

    # training
    print('training...')
    resultFile = open('result/ContextPOSLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
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

        input_pos = Input(batch_shape=(batch_size, posEmbLength,))
        embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)

        embedding_comb = concatenate([embedding_tweet, embedding_pos])
        lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_comb)
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
        places_test = places[test_index]

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
        sampleFile = open('result/ContextPOSLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/ContextPOSLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processContextTLSTM(modelName, balancedWeight='None', embedding='None', char=False, hashtag=True, epochs=4):
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
    dayVector = []
    hourVector = []
    labelCount = {}
    idTagMapper = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
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
                    dayVector.append(np.full((tweetLength), day, dtype='int'))
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    hourVector.append(np.full((tweetLength), hour, dtype='int'))
                    labels.append(activity)
                    if activity not in labelCount:
                        labelCount[activity] = 0.0
                    labelCount[activity] += 1.0
            tweetFile.close()

    dayVector = np.array(dayVector)
    hourVector = np.array(hourVector)

    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/ContextTLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
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
    resultFile = open('result/ContextTLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
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

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        day_train = dayVector[train_index]
        hour_train = hourVector[train_index]

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        day_test = dayVector[test_index]
        hour_test = hourVector[test_index]
        contents_test = np.array(contents)[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            day_train = day_train[:-(len(day_train) % batch_size)]
            hour_train = hour_train[:-(len(hour_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            day_test = day_test[:-(len(day_test) % batch_size)]
            hour_test = hour_test[:-(len(hour_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit([tweet_train, day_train, hour_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, day_train, hour_train], labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit([tweet_train, day_train, hour_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, day_test, hour_test], labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate([tweet_test, day_test, hour_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, day_test, hour_test], batch_size=batch_size)
        sampleFile = open('result/ContextTLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/ContextTLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processContextPOSTLSTM(modelName, balancedWeight='None', embedding='None', char=False, posMode='all', hashtag=True, epochs=4):
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
    dayVector = []
    hourVector = []
    idTagMapper = {}
    places = []
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
                content, pos = extractPOS(idTagMapper[id], posMode)
                contents.append(content)
                posList.append(pos)
                dateTemp = data['created_at'].split()
                day = dayMapper[dateTemp[0]]
                dayVector.append(np.full((tweetLength), day, dtype='int'))
                hour = hourMapper(dateTemp[3].split(':')[0])
                hourVector.append(np.full((tweetLength), hour, dtype='int'))
                labels.append(activity)
                places.append(place)
        tweetFile.close()

    dayVector = np.array(dayVector)
    hourVector = np.array(hourVector)

    places = np.array(places)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open('result/ContextPOSTLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

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
    resultFile = open('result/ContextPOSTLSTM_' + modelName + '_' + balancedWeight + '.result', 'a')
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

        tweet_train = tweetVector[train_index]
        labels_train = labels[train_index]
        day_train = dayVector[train_index]
        hour_train = hourVector[train_index]
        posVector_train = posVector[train_index]

        tweet_test = tweetVector[test_index]
        labels_test = labels[test_index]
        day_test = dayVector[test_index]
        hour_test = hourVector[test_index]
        contents_test = np.array(contents)[test_index]
        places_test = places[test_index]
        posVector_test = posVector[test_index]

        if len(labels_train) % batch_size != 0:
            tweet_train = tweet_train[:-(len(tweet_train) % batch_size)]
            labels_train = labels_train[:-(len(labels_train) % batch_size)]
            day_train = day_train[:-(len(day_train) % batch_size)]
            hour_train = hour_train[:-(len(hour_train) % batch_size)]
            posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
        if len(labels_test) % batch_size != 0:
            tweet_test = tweet_test[:-(len(tweet_test) % batch_size)]
            labels_test = labels_test[:-(len(labels_test) % batch_size)]
            day_test = day_test[:-(len(day_test) % batch_size)]
            hour_test = hour_test[:-(len(hour_test) % batch_size)]
            places_test = places_test[:-(len(places_test) % batch_size)]
            posVector_test = posVector_test[:-(len(posVector_test) % batch_size)]

        labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))
        labelVector_test = np_utils.to_categorical(encoder.transform(labels_test))

        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', labels_train)
            model.fit([tweet_train, posVector_train, day_train, hour_train], labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=0)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
            model.fit([tweet_train, posVector_train, day_train, hour_train], labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=0)
        else:
            model.fit([tweet_train, posVector_train, day_train, hour_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test, day_test, hour_test], labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate([tweet_test, posVector_test, day_test, hour_test], labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict([tweet_test, posVector_test, day_test, hour_test], batch_size=batch_size)
        sampleFile = open('result/ContextPOSTLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/ContextPOSTLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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


def processContextHistLSTM(modelName, balancedWeight='None', embedding='None', char=False, hashtag=True, histNum=1, epochs=7):
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
                # dateTemp = data['created_at'].split()
                # day = dayMapper[dateTemp[0]]
                # hour = hourMapper(dateTemp[3].split(':')[0])
                if id in histMap:
                    for i in range(histNum):
                        histContent = cleanContent(histMap[id][i], hashtag=False, breakEmoji=True)
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
    labelFile = open('result/ContextHistLSTM_' + modelName + '_' + balancedWeight + '.label', 'a')
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
    resultFile = open('result/ContextHistLSTM.' + modelName + '_' + balancedWeight + '.result', 'a')
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
            model.fit(trainList, labelVector_train, epochs=epochs, validation_data=(testList, labelVector_test), batch_size=batch_size, verbose=2)

        scores = model.evaluate(testList, labelVector_test, batch_size=batch_size, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        score += scores[1] * 100
        predictions = model.predict(testList, batch_size=batch_size)
        sampleFile = open('result/ContextHistLSTM_' + modelName + '_' + balancedWeight + '.sample', 'a')
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
    confusionFile = open('result/ContextHistLSTM_' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
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




if __name__ == "__main__":
    #processContextPOSLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=6)

    processContextHistLSTM('long1.5', 'none', 'glove', char=False, hashtag=False, histNum=4, epochs=9)
    #processContextPOSTLSTM('long1.5', 'none', 'glove', char=False, posMode='all', hashtag=False, epochs=6)
