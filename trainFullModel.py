from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Merge, Input, concatenate, Lambda
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from utilities import word2vecReader
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import math, pickle, json, sys
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

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}

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


def loadHistData(modelName, histName, char, embedding, resultName, histNum=5):
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
    for i in range(histNum):
        histContents_train[i] = []
        histDayVectors_train[i] = []
        histHourVectors_train[i] = []
        histPOSLists_train[i] = []

    contents_train = []
    labels_train = []
    places_train = []
    days_train = []
    hours_train = []
    poss_train = []
    ids_train = []
    inputFileList = ['data/consolidateData_' + modelName + '_train.json', 'data/consolidateData_' + modelName + '_dev.json', 'data/consolidateData_' + modelName + '_test.json']
    for inputFilename in inputFileList:
        inputFile = open(inputFilename, 'r')
        for line in inputFile:
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
        inputFile.close()

    for i in range(histNum):
        histDayVectors_train[i] = np.array(histDayVectors_train[i])
        histHourVectors_train[i] = np.array(histHourVectors_train[i])
    days_train = np.array(days_train)
    hours_train = np.array(hours_train)
    places_train = np.array(places_train)
    ids_train = np.array(ids_train)

    if char:
        tk = Tokenizer(num_words=vocabSize, char_level=char, filters='')
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)

    totalList = contents_train[:]
    for i in range(histNum):
        totalList += histContents_train[i]
    tk.fit_on_texts(totalList)
    tweetSequences_train = tk.texts_to_sequences(contents_train)
    tweetVector_train = sequence.pad_sequences(tweetSequences_train, maxlen=tweetLength, truncating='post', padding='post')

    with open(resultName + '_tweet.tk', 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    histTweetVectors_train = []
    for i in range(histNum):
        histSequence_train = tk.texts_to_sequences(histContents_train[i])
        tempVector_train = sequence.pad_sequences(histSequence_train, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors_train.append(tempVector_train)

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

    posVocabSize = 25
    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    totalPOSList = poss_train[:]
    for i in range(histNum):
        totalPOSList += histPOSLists_train[i]
    tkPOS.fit_on_texts(totalPOSList)

    posSequences_train = tkPOS.texts_to_sequences(poss_train)
    posVector_train = sequence.pad_sequences(posSequences_train, maxlen=tweetLength, truncating='post', padding='post')

    with open(resultName + '_pos.tk', 'wb') as handle:
        pickle.dump(tkPOS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    histPOSVectors_train = []
    for i in range(histNum):
        histPOSSequences_train = tkPOS.texts_to_sequences(histPOSLists_train[i])
        histPOSVector_train = sequence.pad_sequences(histPOSSequences_train, maxlen=tweetLength, truncating='post', padding='post')
        histPOSVectors_train.append(histPOSVector_train)

    return ids_train, labels_train, places_train, contents_train, days_train, hours_train, poss_train, tweetVector_train, posVector_train, histTweetVectors_train, histDayVectors_train, histHourVectors_train, histPOSVectors_train, posVocabSize, embMatrix, word_index


def trainLSTM(modelName, balancedWeight='None', char=False, epochs=4):
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()
    activityList = []
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for line in activityListFile:
        if not line.startswith('#'):
            activityList.append(line.strip())
    activityListFile.close()
    labelNum = len(np.unique(activityList))
    if 'NONE' in activityList:
        labelNum -= 1

    contents = []
    labels = []
    timeList = []
    labelTweetCount = {}
    placeTweetCount = {}
    labelCount = {}
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
                    contents.append(data['text'].encode('utf-8'))
                    dateTemp = data['created_at'].split()
                    timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
                    if activity not in labelCount:
                        labelCount[activity] = 1.0
                    else:
                        labelCount[activity] += 1.0
                    labels.append(activity)
                    tweetCount += 1
            tweetFile.close()
            labelTweetCount[activity] += tweetCount
            placeTweetCount[place] = tweetCount
    activityLabels = np.array(labels)
    timeVector = np.array(timeList)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelFile = open('model/LSTM_'+modelName + '_' + str(balancedWeight) + '.label', 'w')
    labelFile.write(str(encoder.classes_).replace('\n', ' ').replace("'", "")[1:-1].replace(' ', '\t'))
    labelFile.close()
    encodedLabels = encoder.transform(labels)
    labels = np_utils.to_categorical(encodedLabels)
    labelList = encoder.classes_.tolist()

    tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    pickle.dump(tk, open('model/LSTM_' + modelName + '_' + str(balancedWeight) + '.tk', 'wb'))
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, padding='post', truncating='post')

    model_text = Sequential()
    model_text.add(Embedding(vocabSize, embeddingVectorLength))
    model_text.add(Dropout(0.2))
    model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

    model_time = Sequential()
    model_time.add(Dense(2, input_shape=(2,), activation='relu'))

    model = Sequential()
    model.add(Merge([model_text, model_time], mode='concat'))
    model.add(Dense(labelNum, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels)
        model.fit([tweetVector, timeVector], labels, epochs=epochs, batch_size=10, sample_weight=sampleWeight)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(activityLabels), activityLabels)
        model.fit([tweetVector, timeVector], labels, epochs=epochs, batch_size=10, class_weight=classWeight, verbose=1)
    elif balancedWeight == 'class_label':
        classWeight = []
        countSum = sum(labelCount.values())
        for label in labelList:
            classWeight.append(countSum/labelCount[label])
        model.fit([tweetVector, timeVector], labels, epochs=epochs, batch_size=10, class_weight=classWeight)
    elif balancedWeight == 'class_label_log':
        classWeight = []
        countSum = sum(labelCount.values())
        for label in labelList:
            classWeight.append(-math.log(labelCount[label] / countSum))
        model.fit([tweetVector, timeVector], labels, epochs=epochs, batch_size=10, class_weight=classWeight)
    else:
        model.fit([tweetVector, timeVector], labels, epochs=epochs, batch_size=10)

    model_json = model.to_json()
    with open('model/LSTM_'+modelName + '_' + str(balancedWeight) + '.json', 'w') as modelFile:
        modelFile.write(model_json)
    model.save_weights('model/LSTM_' + modelName + '_' + str(balancedWeight) + '.h5')


def trainHybridLSTM(modelName, histName, balancedWeight='None', embedding='glove', char=False, histNum=5, epochs=7):
    resultName = 'model/J-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight
    ids_train, labels_train, places_train, contents_train, days_train, hours_train, poss_train, tweetVector_train, posVector_train, histTweetVectors_train, histDayVectors_train, \
    histHourVectors_train, histPOSVectors_train, posVocabSize, embMatrix, word_index = loadHistData(modelName, histName, char, embedding, resultName=resultName, histNum=histNum)

    labelNum = len(np.unique(labels_train))
    encoder = LabelEncoder()
    encoder.fit(labels_train)
    labels_train = encoder.transform(labels_train)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    print('training...')
    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    shared_embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
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

    tweet_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
    labels_train = labels_train[:-(len(labels_train) % batch_size)]
    days_train = days_train[:-(len(days_train) % batch_size)]
    hours_train = hours_train[:-(len(hours_train) % batch_size)]
    posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
    for i in range(histNum):
        histTweetVectors_train[i] = histTweetVectors_train[i][:-(len(histTweetVectors_train[i]) % batch_size)]
        histDayVectors_train[i] = histDayVectors_train[i][:-(len(histDayVectors_train[i]) % batch_size)]
        histHourVectors_train[i] = histHourVectors_train[i][:-(len(histHourVectors_train[i]) % batch_size)]
        histPOSVectors_train[i] = histPOSVectors_train[i][:-(len(histPOSVectors_train[i]) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)

    trainList = [tweet_train, days_train, hours_train, posVector_train]
    for i in range(histNum):
        trainList += [histTweetVectors_train[i], histDayVectors_train[i], histHourVectors_train[i], histPOSVectors_train[i]]

    verbose = 1
    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    model_json = model.to_json()
    with open(resultName+'_model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(resultName+'_model.h5')

    print('FINSIHED')


def trainHybridAttLSTM(modelName, histName, balancedWeight='None', embedding='glove', char=False, histNum=5, epochs=7):
    resultName = 'model/J-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight
    ids_train, labels_train, places_train, contents_train, days_train, hours_train, poss_train, tweetVector_train, posVector_train, histTweetVectors_train, histDayVectors_train, \
    histHourVectors_train, histPOSVectors_train, posVocabSize, embMatrix, word_index = loadHistData(modelName, histName, char, embedding, resultName=resultName, histNum=histNum)

    labelNum = len(np.unique(labels_train))
    encoder = LabelEncoder()
    encoder.fit(labels_train)
    labels_train = encoder.transform(labels_train)
    labelList = encoder.classes_.tolist()
    print('Labels: ' + str(labelList))
    labelFile = open(resultName + '.label', 'a')
    labelFile.write(str(labelList) + '\n')
    labelFile.close()

    # training
    print('training...')
    input_tweet = Input(batch_shape=(batch_size, tweetLength,), name='tweet_input')
    shared_embedding_tweet = Embedding(len(word_index) + 1, 200, weights=[embMatrix], trainable=True)
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
    tweet_lstm = LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(comb)
    self_attention = SeqSelfAttention(attention_activation='sigmoid')(tweet_lstm)
    last_timestep = Lambda(lambda x: x[:, -1, :])(self_attention)

    conList = [last_timestep]
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
        lstm_temp = LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(comb_temp)
        self_attention_temp = SeqSelfAttention(attention_activation='sigmoid')(lstm_temp)
        last_timestep_temp = Lambda(lambda x: x[:, -1, :])(self_attention_temp)
        conList.append(last_timestep_temp)
        inputList += [input_hist, input_day_temp, input_hour_temp, input_pos_temp]

    comb_total = concatenate(conList)
    output = Dense(labelNum, activation='softmax', name='output')(comb_total)
    model = Model(inputs=inputList, outputs=output)
    #print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    tweet_train = tweetVector_train[:-(len(tweetVector_train) % batch_size)]
    labels_train = labels_train[:-(len(labels_train) % batch_size)]
    days_train = days_train[:-(len(days_train) % batch_size)]
    hours_train = hours_train[:-(len(hours_train) % batch_size)]
    posVector_train = posVector_train[:-(len(posVector_train) % batch_size)]
    for i in range(histNum):
        histTweetVectors_train[i] = histTweetVectors_train[i][:-(len(histTweetVectors_train[i]) % batch_size)]
        histDayVectors_train[i] = histDayVectors_train[i][:-(len(histDayVectors_train[i]) % batch_size)]
        histHourVectors_train[i] = histHourVectors_train[i][:-(len(histHourVectors_train[i]) % batch_size)]
        histPOSVectors_train[i] = histPOSVectors_train[i][:-(len(histPOSVectors_train[i]) % batch_size)]

    labelVector_train = np_utils.to_categorical(labels_train)

    trainList = [tweet_train, days_train, hours_train, posVector_train]
    for i in range(histNum):
        trainList += [histTweetVectors_train[i], histDayVectors_train[i], histHourVectors_train[i], histPOSVectors_train[i]]

    verbose = 1
    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels_train)
        model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, sample_weight=sampleWeight, verbose=verbose)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(labels_train), labels_train)
        model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, class_weight=classWeight, verbose=verbose)
    else:
        model.fit(trainList, labelVector_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

    model_json = model.to_json()
    with open(resultName+'_model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(resultName+'_model.h5')

    print('FINSIHED')



if __name__ == '__main__':
    #trainLSTM('long1.5', 'none', char=False)
    #trainHybridLSTM('long1.5', 'long1.5', 'class', 'glove', char=False, histNum=5, epochs=26)
    trainHybridAttLSTM('long1.5', 'long1.5', 'class', 'glove', char=False, histNum=5, epochs=14)
