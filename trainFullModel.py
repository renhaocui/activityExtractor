from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Merge, Input, concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import math, pickle, json, sys
reload(sys)
sys.setdefaultencoding('utf8')

vocabSize = 10000
tweetLength = 25
posEmbLength = 25
embeddingVectorLength = 200
embeddingPOSVectorLength = 20
charLengthLimit = 20
batch_size = 10

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


def loadHistData(modelName, char, histNum=5, resultName=''):
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

    with open(resultName+'_tweet.tk', 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)

    histTweetVectors = []
    for i in range(histNum):
        histSequence = tk.texts_to_sequences(histContents[i])
        tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
        histTweetVectors.append(tempVector)

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

    posVocabSize = 25
    tkPOS = Tokenizer(num_words=posVocabSize, filters='', lower=False)
    totalPOSList = poss[:]
    for i in range(histNum):
        totalPOSList += histPOSLists[i]
    tkPOS.fit_on_texts(totalPOSList)
    posSequences = tkPOS.texts_to_sequences(poss)
    posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

    with open(resultName + '_pos.tk', 'wb') as handle:
        pickle.dump(tkPOS, handle, protocol=pickle.HIGHEST_PROTOCOL)

    histPOSVectors = []
    for i in range(histNum):
        histPOSSequences = tkPOS.texts_to_sequences(histPOSLists[i])
        histPOSVector = sequence.pad_sequences(histPOSSequences, maxlen=tweetLength, truncating='post', padding='post')
        histPOSVectors.append(histPOSVector)

    return ids, labels, places, contents, days, hours, poss, tweetVector, posVector, histTweetVectors, histDayVectors, histHourVectors, histPOSVectors, posVocabSize, embMatrix, word_index


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


def trainHybridLSTM(modelName, balancedWeight='None', char=False, histNum=1, epochs=7):
    resultName = 'result/model/C-Hist-Context-POST-LSTM_' + modelName + '_' + balancedWeight
    ids, labels, places, contents, dayVector, hourVector, posList, tweetVector, posVector, histTweetVectors, histDayVector, histHourVector, histPOSVectors, posVocabSize, embMatrix, word_index = loadHistData(
        modelName, char, histNum=histNum, resultName=resultName)
    labelNum = len(np.unique(labels))
    labels = np.array(labels)
    encoder = LabelEncoder()
    encoder.fit(labels)
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

    histTweetVector_train = []
    histDayVector_train = []
    histHourVector_train = []
    histPOSVector_train = []

    tweet_train = tweetVector[:-(len(tweetVector) % batch_size)]
    labels_train = labels[:-(len(labels) % batch_size)]
    day_train = dayVector[:-(len(dayVector) % batch_size)]
    hour_train = hourVector[:-(len(hourVector) % batch_size)]
    pos_train = posVector[:-(len(posVector) % batch_size)]
    for i in range(histNum):
        histTweetVector_train.append(histTweetVectors[i][:-(len(histTweetVectors[i]) % batch_size)])
        histDayVector_train.append(histDayVector[i][:-(len(histDayVector[i]) % batch_size)])
        histHourVector_train.append(histHourVector[i][:-(len(histHourVector[i]) % batch_size)])
        histPOSVector_train.append(histPOSVectors[i][:-(len(histPOSVectors[i]) % batch_size)])

    labelVector_train = np_utils.to_categorical(encoder.transform(labels_train))

    trainList = [tweet_train, day_train, hour_train, pos_train]
    for i in range(histNum):
        trainList += [histTweetVector_train[i], histDayVector_train[i], histHourVector_train[i], histPOSVector_train[i]]

    verbose=0
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

    print('DONE')



if __name__ == '__main__':
    #trainLSTM('long1.5', 'none', char=False)
    trainHybridLSTM('long1.5', 'none', char=False, histNum=5, epochs=4)
