import json
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, Merge, Input, multiply
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from keras.utils import to_categorical
from utilities import word2vecReader

vocabSize = 10000
tweetLength = 25
posEmbLength = 25
embeddingVectorLength = 200
embeddingPOSVectorLength = 200
charLengthLimit = 20

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}
POSMapper = {'N': 'N', 'O': 'N', '^': 'N', 'S': 'N', 'Z': 'N', 'L': 'N', 'M': 'N',
             'V': 'V', 'A': 'A', 'R': 'R', '@': '@', '#': '#', '~': '~', 'U': 'U', 'E': 'E', ',': ',',
             '!': '0', 'D': '0', 'P': '0', '&': '0', 'T': '0', 'X': '0', 'Y': '0', '$': '0', 'G': '0'}

def extractPOS(inputList, mode):
    output = ''
    for item in inputList:
        if mode == 'all':
            output += item[1] + ' '
        else:
            output += POSMapper[item[1]] + ' '
    return output.strip()

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


def processLSTM(modelName, char=False, epochs=3):
    activityList = []
    activityListFile = open('lists/google_place_activity_'+modelName+'.list', 'r')
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
    dayList = []
    hourList = []
    labelTweetCount = {}
    placeTweetCount = {}
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
                    contents.append(data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower())
                    dateTemp = data['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    dayList.append(day)
                    hourList.append(hour)
                    labels.append(activity)
                    tweetCount += 1
                    if activity not in labelCount:
                        labelCount[activity] = 1.0
                    else:
                        labelCount[activity] += 1.0
            tweetFile.close()
            labelTweetCount[activity] += tweetCount
            placeTweetCount[place] = tweetCount
    activityLabels = np.array(labels)
    dayVector = to_categorical(dayList, num_classes=7)
    hourVector = to_categorical(hourList, num_classes=4)
    encoder = LabelEncoder()
    encoder.fit(labels)
    encodedLabels = encoder.transform(labels)
    labels = np_utils.to_categorical(encodedLabels)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')
    if char:
        oneHotVector = []
        for tweet in tweetVector:
            oneHotVector.append(to_categorical(tweet, num_classes=70))
        tweetVector = np.array(oneHotVector)

    # training
    print('training...')
    tweet_train, tweet_test, day_train, day_test, hour_train, hour_test, label_train, label_test, activityLabels_train, activityLabels_test = train_test_split(tweetVector, dayVector, hourVector, labels, activityLabels)
    model_text = Sequential()
    if char:
        model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(140, 70)))
    else:
        model_text.add(Embedding(vocabSize, embeddingVectorLength))
        model_text.add(Dropout(0.2))
        model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model_time1 = Sequential()
    model_time1.add(Dense(1, input_shape=(4,), activation='relu', name='hour'))

    model_time2 = Sequential()
    model_time2.add(Dense(1, input_shape=(7,), activation='relu', name='day'))

    model = Sequential()
    model.add(Merge([model_text, model_time1, model_time2], mode='concat'))
    model.add(Dense(labelNum, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    resultFile = open('result/resultEpoch', 'a')
    for epoch in range(epochs):
        model.fit([tweet_train, hour_train, day_train], label_train, epochs=1, batch_size=10, verbose=0)
        scores = model.evaluate([tweet_test, hour_test, day_test], label_test, verbose=0)
        print('Epoch '+str(epoch)+': '+str(scores[1]*100))
        resultFile.write('Epoch '+str(epoch)+': '+str(scores[1]*100)+'\n\n')
    resultFile.close()


def processSLSTM(modelName, char=False, posMode='all', epochs=1):
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
                    #content = cleaner.tweetCleaner(content)
                    contents.append(content)
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
    tweet_train, tweet_test, day_train, day_test, hour_train, hour_test, posVector_train, posVector_test, label_train, label_test = train_test_split(
        tweetVector, dayVector, hourVector, posVector, labels)

    labelVector_train = np_utils.to_categorical(encoder.transform(label_train))
    labelVector_test = np_utils.to_categorical(encoder.transform(label_test))

    model_text = Sequential()
    model_text.add(Embedding(vocabSize, embeddingVectorLength))
    model_text.add(Dropout(0.2))
    model_text.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))

    model_time1 = Sequential()
    model_time1.add(Dense(1, input_shape=(4,), activation='relu', name='hour'))

    model_time2 = Sequential()
    model_time2.add(Dense(1, input_shape=(7,), activation='relu', name='day'))

    model_pos = Sequential()
    model_pos.add(Embedding(posVocabSize, posEmbLength))
    model_pos.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))

    model = Sequential()
    model.add(Merge([model_text, model_time1, model_time2, model_pos], mode='concat'))
    model.add(Dense(labelNum, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    resultFile = open('result/resultEpoch', 'a')
    for epoch in range(epochs):
        model.fit([tweet_train, hour_train, day_train, posVector_train], labelVector_train, epochs=epochs, batch_size=10, verbose=0)
        scores = model.evaluate([tweet_test, hour_test, day_test, posVector_test], labelVector_test, batch_size=1, verbose=0)
        print('Epoch ' + str(epoch) + ': ' + str(scores[1] * 100))
        resultFile.write('Epoch ' + str(epoch) + ': ' + str(scores[1] * 100) + '\n\n')
    resultFile.close()


def processHLSTM(modelName, char=False, posMode='all', epochs=1):
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
                    #content = cleaner.tweetCleaner(content)
                    contents.append(content)
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
    tweet_train, tweet_test, day_train, day_test, hour_train, hour_test, posVector_train, posVector_test, label_train, label_test = train_test_split(
        tweetVector, dayVector, hourVector, posVector, labels)

    labelVector_train = np_utils.to_categorical(encoder.transform(label_train))
    labelVector_test = np_utils.to_categorical(encoder.transform(label_test))

    model_text = Sequential()
    model_text.add(Embedding(vocabSize, embeddingVectorLength))
    model_text.add(Dropout(0.2))
    model_text.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))

    model_time1 = Sequential()
    model_time1.add(Dense(1, input_shape=(4,), activation='relu', name='hour'))

    model_time2 = Sequential()
    model_time2.add(Dense(1, input_shape=(7,), activation='relu', name='day'))

    model_pos = Sequential()
    model_pos.add(Embedding(posVocabSize, posEmbLength))
    model_pos.add(LSTM(25, dropout=0.2, recurrent_dropout=0.2))

    model = Sequential()
    model.add(Merge([model_text, model_time1, model_time2, model_pos], mode='concat'))
    model.add(Dense(labelNum, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    resultFile = open('result/resultEpoch', 'a')
    for epoch in range(epochs):
        model.fit([tweet_train, hour_train, day_train, posVector_train], labelVector_train, epochs=epochs, batch_size=10, verbose=0)
        scores = model.evaluate([tweet_test, hour_test, day_test, posVector_test], labelVector_test, batch_size=1, verbose=0)
        print('Epoch ' + str(epoch) + ': ' + str(scores[1] * 100))
        resultFile.write('Epoch ' + str(epoch) + ': ' + str(scores[1] * 100) + '\n\n')
    resultFile.close()


def processTLSTM(modelName, char=False, epochs=3):
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
    dayList = []
    hourList = []
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            tweetFile = open('data/POIplace/' + place + '.json', 'r')
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                    contents.append(content)
                    dateTemp = data['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    dayList.append(day)
                    hourList.append(hour)
                    labels.append(activity)
            tweetFile.close()

    dayVector = to_categorical(dayList, num_classes=7)
    hourVector = to_categorical(hourList, num_classes=4)
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

    tweet_train, tweet_test, day_train, day_test, hour_train, hour_test, label_train, label_test = train_test_split(tweetVector, dayVector, hourVector, labels)
    labelVector_train = np_utils.to_categorical(encoder.transform(label_train))
    labelVector_test = np_utils.to_categorical(encoder.transform(label_test))

    model_text = Sequential()
    model_text.add(Embedding(vocabSize, embeddingVectorLength))
    model_text.add(Dropout(0.2))
    model_text.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2))

    model_time1 = Sequential()
    model_time1.add(Dense(5, input_shape=(4,), activation='relu', name='hour'))

    model_time2 = Sequential()
    model_time2.add(Dense(5, input_shape=(7,), activation='relu', name='day'))

    model = Sequential()
    model.add(Merge([model_text, model_time1, model_time2], mode='concat'))
    model.add(Dense(labelNum, activation='softmax', name='output'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    resultFile = open('result/resultEpoch', 'a')
    for epoch in range(epochs):
        model.fit([tweet_train, hour_train, day_train], labelVector_train, epochs=epochs, batch_size=10, verbose=0)
        scores = model.evaluate([tweet_test, hour_test, day_test], labelVector_test, batch_size=1, verbose=0)
        print('Epoch ' + str(epoch) + ': ' + str(scores[1] * 100))
        resultFile.write('Epoch ' + str(epoch) + ': ' + str(scores[1] * 100) + '\n\n')
    resultFile.close()


def processMLSTM(modelName, embedding='None', char=False, posMode='all', epochs=4):
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
                    # content = cleaner.tweetCleaner(content)
                    contents.append(content)
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

    tweet_train, tweet_test, posVector_train, posVector_test, label_train, label_test = train_test_split(tweetVector, posVector, labels)
    labelVector_train = np_utils.to_categorical(encoder.transform(label_train))
    labelVector_test = np_utils.to_categorical(encoder.transform(label_test))

    input_tweet = Input(shape=(tweetLength,), name='tweet_input')
    if embedding in ['glove', 'word2vec']:
        Embedding(len(word_index) + 1, embeddingVectorLength, weights=[embMatrix], trainable=False)(input_tweet)
    else:
        embedding_tweet = Embedding(vocabSize, embeddingVectorLength)(input_tweet)
    tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_tweet)

    input_day = Input(shape=(7,))
    day_dense = Dense(1, activation='relu', name='day')(input_day)

    input_hour = Input(shape=(4,))
    hour_dense = Dense(1, activation='relu', name='hour')(input_hour)

    input_pos = Input(shape=(posEmbLength,))
    embedding_pos = Embedding(posVocabSize, embeddingPOSVectorLength)(input_pos)
    pos_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2)(embedding_pos)

    comb = multiply([tweet_lstm, pos_lstm])
    #comb = concatenate([tweet_lstm, hour_dense, day_dense, pos_lstm])
    output = Dense(labelNum, activation='softmax', name='output')(comb)
    model = Model(inputs=[input_tweet, input_pos], outputs=output)
    #print model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit([tweet_train, posVector_train], labelVector_train, epochs=epochs, validation_data=([tweet_test, posVector_test], labelVector_test), batch_size=10,
                  verbose=2)




if __name__ == '__main__':
    #processLSTM('long1.5', char=False, epochs=10)
    #processTLSTM('long1.5', char=False, epochs=10)
    #processSLSTM('long1.5', char=False, posMode='map', epochs=5)
    processMLSTM('long1.5', 'none', char=False, posMode='all', epochs=10)
