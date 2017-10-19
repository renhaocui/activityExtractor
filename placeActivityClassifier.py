import json
import math
import pickle

import numpy as np
import word2vecReader
from keras.layers import Dense, LSTM, Dropout, Merge, MaxPooling1D, Conv1D, Flatten, Reshape, InputLayer
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from utilities.tokenizer import simpleTokenize

vocabSize = 70
tweetLength = 140
embeddingVectorLength = 200
EMBEDDING_word2vec = 400
EMBEDDING_tweet2vec = 500
charLengthLimit = 30

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}

excludeList = {}

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


def contents2vecs(model, contents):
    outEmb = []
    invalidList = []
    for index, content in enumerate(contents):
        words = simpleTokenize(content)
        tempList = []
        for word in words:
            if word in model.vocab:
                tempList.append(model[word])
        if len(tempList) < 1:
            invalidList.append(index)
            #output = None
        else:
            vecSize = len(tempList[0])
            sumList = []
            for i in range(vecSize):
                sumList.append(0.0)
            for vec in tempList:
                for i in range(vecSize):
                    sumList[i] += vec[i]
            output = []
            dataSize = len(tempList)
            for value in sumList:
                output.append(value/dataSize)
            outEmb.append(output)
    return np.array(outEmb), invalidList


def words2vecs(model, contents, tweetLength):
    invalidList = []
    outEmb = []
    for index, content in enumerate(contents):
        words = simpleTokenize(content)
        wordEmbList = []
        validCount = 0
        for i in range(tweetLength):
            if index < len(words):
                word = words[index]
                if word in model.vocab:
                    wordEmbList.append(model[word])
                    validCount += 1
        if validCount == 0:
            invalidList.append(index)
        else:
            outEmb.append(wordEmbList)
    return np.array(outEmb), invalidList


def processCNNLSTM(labelNum, modelName, balancedWeight='None', clean=False, char=False):
    activityList = []
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for line in activityListFile:
        if not line.startswith('#'):
            if line.strip() in excludeList:
                activityList.append('NONE')
            else:
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
    labelCount = {}
    timeList = []
    labelTweetCount = {}
    placeTweetCount = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            if activity not in labelTweetCount:
                labelTweetCount[activity] = 0.0
            if clean:
                tweetFile = open('data/google_place_tweets3.3/POI_clean/' + place + '.json', 'r')
            else:
                tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
            tweetCount = 0
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    contents.append(data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8'))
                    dateTemp = data['created_at'].split()
                    timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
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
    timeVector = np.array(timeList)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print labelList
    encodedLabels = encoder.transform(labels)
    labels = np_utils.to_categorical(encodedLabels)

    if embedding == 'word2vec':
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tweetVector, invalidList = contents2vecs(embModel, contents)
        # tweetVector, invalidList = words2vecs(embModel, contents, tweetLength)
        timeVector = np.delete(timeVector, invalidList, 0)
        labels = np.delete(labels, invalidList, 0)
        activityLabels = np.delete(activityLabels, invalidList, 0)
        tweetVector = np.reshape(tweetVector, (tweetVector.shape[0], 1, tweetVector.shape[1]))
    elif embedding == 'gensim':
        tk = Tokenizer(num_words=vocabSize, char_level=char)
        tk.fit_on_texts(contents)
        tweetSequences = tk.texts_to_sequences(contents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index) + 1, EMBEDDING_word2vec))
        for word, i in word_index.items():
            if word in embModel:
                embVector = embModel[word]
                embMatrix[i] = embVector
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
        tk.fit_on_texts(contents)
        tweetSequences = tk.texts_to_sequences(contents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)

    # training
    print('training...')
    resultFile = open('result/result', 'a')
    accuracy = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(timeVector, activityLabels)):
        model_text = Sequential()
        model_text.add(
            Embedding(input_dim=vocabSize, output_dim=embeddingVectorLength, input_length=tweetLength, name='emb'))
        model_text.add(Dropout(0.2))
        model_text.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, name='LSTM'))

        # day & time input
        model_time = Sequential()
        model_time.add(Dense(2, input_shape=(2,), activation='relu', name='time_input'))
        # merge text and time branches
        model = Sequential()
        model.add(Merge([model_text, model_time], mode='concat', name='finalMerge'))
        # model.add(Dropout(0.5))
        model.add(Dense(labelNum, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        tweet_train = tweetVector[train_index]
        time_train = timeVector[train_index]
        label_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        time_test = timeVector[test_index]
        label_test = labels[test_index]
        activityLabels_train = activityLabels[train_index]
        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', label_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, sample_weight=sampleWeight)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(activityLabels_train), activityLabels_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(countSum / labelCount[label])
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label_log':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(-math.log(labelCount[label] / countSum))
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        else:
            model.fit([tweet_train, time_train], label_train,
                      validation_data=([tweet_test, time_test], label_test), epochs=3, batch_size=10)

        scores = model.evaluate([tweet_test, time_test], label_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        resultFile.write('Fold ' + str(fold) + ': ' + str(scores[1] * 100) + '\n')
        accuracy += scores[1] * 100
    resultFile.write('Overall: ' + str(accuracy / 5) + '\n\n')
    print(accuracy / 5)
    resultFile.close()


def processAttLSTM(labelNum, modelName, balancedWeight='None', clean=False, char=False):
    activityList = []
    activityListFile = open('lists/google_place_activity_'+modelName+'.list', 'r')
    for line in activityListFile:
        if not line.startswith('#'):
            if line.strip() in excludeList:
                activityList.append('NONE')
            else:
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
    labelCount = {}
    timeList = []
    labelTweetCount = {}
    placeTweetCount = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity != 'NONE':
            if activity not in labelTweetCount:
                labelTweetCount[activity] = 0.0
            if clean:
                tweetFile = open('data/google_place_tweets3.3/POI_clean/' + place + '.json', 'r')
            else:
                tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
            tweetCount = 0
            for line in tweetFile:
                data = json.loads(line.strip())
                if len(data['text']) > charLengthLimit:
                    contents.append(data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8'))
                    dateTemp = data['created_at'].split()
                    timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
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
    timeVector = np.array(timeList)
    encoder = LabelEncoder()
    encoder.fit(labels)
    labelList = encoder.classes_.tolist()
    print labelList
    encodedLabels = encoder.transform(labels)
    labels = np_utils.to_categorical(encodedLabels)

    if embedding == 'word2vec':
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        tweetVector, invalidList = contents2vecs(embModel, contents)
        #tweetVector, invalidList = words2vecs(embModel, contents, tweetLength)
        timeVector = np.delete(timeVector, invalidList, 0)
        labels = np.delete(labels, invalidList, 0)
        activityLabels = np.delete(activityLabels, invalidList, 0)
        tweetVector = np.reshape(tweetVector, (tweetVector.shape[0], 1, tweetVector.shape[1]))
    elif embedding == 'gensim':
        tk = Tokenizer(num_words=vocabSize, char_level=char)
        tk.fit_on_texts(contents)
        tweetSequences = tk.texts_to_sequences(contents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
        word_index = tk.word_index
        w2v = word2vecReader.Word2Vec()
        embModel = w2v.loadModel()
        embMatrix = np.zeros((len(word_index) + 1, EMBEDDING_word2vec))
        for word, i in word_index.items():
            if word in embModel:
                embVector = embModel[word]
                embMatrix[i] = embVector
    else:
        tk = Tokenizer(num_words=vocabSize, char_level=char)
        tk.fit_on_texts(contents)
        tweetSequences = tk.texts_to_sequences(contents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, padding='post', truncating='post')

    # training
    print('training...')
    resultFile = open('result/result', 'a')
    accuracy = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(timeVector, activityLabels)):
        model_text1 = Sequential()
        model_text1.add(Embedding(input_dim=vocabSize, output_dim=embeddingVectorLength, input_length=tweetLength, name='emb1'))
        model_text1.add(Dropout(0.2))
        model_text1.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='LSTM1'))

        model_text2 = Sequential()
        model_text2.add(Embedding(input_dim=vocabSize, output_dim=embeddingVectorLength, input_length=tweetLength, name='emb2'))
        model_text2.add(Dropout(0.2))
        model_text2.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, name='LSTM2'))

        model_text0 = Sequential()
        model_text0.add(Merge([model_text1, model_text2], mode='dot', dot_axes=[1, 1], name='merge1'))
        model_text0.add(Flatten())
        model_text0.add(Dense(tweetLength * 100, activation='relu'))
        model_text0.add(Reshape((tweetLength, 100)))

        model_text = Sequential()
        model_text.add(Merge([model_text1, model_text0], mode='sum', name='merge0'))
        model_text.add(Flatten())

        # day & time input
        model_time = Sequential()
        model_time.add(Dense(2, input_shape=(2,), activation='relu', name='time_input'))
        # merge text and time branches
        model = Sequential()
        model.add(Merge([model_text, model_time], mode='concat', name='finalMerge'))
        #model.add(Dropout(0.5))
        model.add(Dense(labelNum, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        tweet_train = tweetVector[train_index]
        time_train = timeVector[train_index]
        label_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        time_test = timeVector[test_index]
        label_test = labels[test_index]
        activityLabels_train = activityLabels[train_index]
        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', label_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, sample_weight=sampleWeight)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(activityLabels_train), activityLabels_train)
            model.fit([tweet_train, tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(countSum/labelCount[label])
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label_log':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(-math.log(labelCount[label]/countSum))
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        else:
            model.fit([tweet_train, tweet_train, time_train], label_train, validation_data=([tweet_test, tweet_test, time_test], label_test), epochs=3, batch_size=10)

        scores = model.evaluate([tweet_test, tweet_test, time_test], label_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        resultFile.write('Fold '+str(fold)+': '+str(scores[1] * 100)+'\n')
        accuracy += scores[1] * 100
    resultFile.write('Overall: '+str(accuracy/5)+'\n\n')
    print(accuracy/5)
    resultFile.close()


def processLSTM(labelNum, modelName, balancedWeight='None', embedding='None', clean=False, char=False):
    if embedding == 'tweet2vec':
        embData = np.load('embModel/embeddings.npy')
        tweetVector = np.reshape(embData, (embData.shape[0], 1, embData.shape[1]))
        print len(embData)
        labels = []
        labelFile = open('embModel/long1out.label', 'r')
        for line in labelFile:
            labels.append(line.strip())
        labelFile.close()
        activityLabels = np.array(labels)
        encoder = LabelEncoder()
        encoder.fit(labels)
        labelList = encoder.classes_.tolist()
        print labelList
        encodedLabels = encoder.transform(labels)
        labels = np_utils.to_categorical(encodedLabels)

        timeList = []
        timeFile = open('embModel/long1out.time', 'r')
        for line in timeFile:
            dateTemp = line.strip().split()
            timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
        timeFile.close()
        timeVector = np.array(timeList)

    else:
        activityList = []
        activityListFile = open('lists/google_place_activity_'+modelName+'.list', 'r')
        for line in activityListFile:
            if not line.startswith('#'):
                if line.strip() in excludeList:
                    activityList.append('NONE')
                else:
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
        labelCount = {}
        timeList = []
        labelTweetCount = {}
        placeTweetCount = {}
        for index, place in enumerate(placeList):
            activity = activityList[index]
            if activity != 'NONE':
                if activity not in labelTweetCount:
                    labelTweetCount[activity] = 0.0
                if clean:
                    tweetFile = open('data/google_place_tweets3.3/POI_clean/' + place + '.json', 'r')
                else:
                    tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
                tweetCount = 0
                for line in tweetFile:
                    data = json.loads(line.strip())
                    if len(data['text']) > charLengthLimit:
                        contents.append(data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8'))
                        dateTemp = data['created_at'].split()
                        timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
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
        timeVector = np.array(timeList)
        encoder = LabelEncoder()
        encoder.fit(labels)
        labelList = encoder.classes_.tolist()
        print labelList
        encodedLabels = encoder.transform(labels)
        labels = np_utils.to_categorical(encodedLabels)

        if embedding == 'word2vec':
            w2v = word2vecReader.Word2Vec()
            embModel = w2v.loadModel()
            tweetVector, invalidList = contents2vecs(embModel, contents)
            #tweetVector, invalidList = words2vecs(embModel, contents, tweetLength)
            timeVector = np.delete(timeVector, invalidList, 0)
            labels = np.delete(labels, invalidList, 0)
            activityLabels = np.delete(activityLabels, invalidList, 0)
            tweetVector = np.reshape(tweetVector, (tweetVector.shape[0], 1, tweetVector.shape[1]))
        elif embedding == 'gensim':
            tk = Tokenizer(num_words=vocabSize, char_level=char)
            tk.fit_on_texts(contents)
            tweetSequences = tk.texts_to_sequences(contents)
            tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
            word_index = tk.word_index
            w2v = word2vecReader.Word2Vec()
            embModel = w2v.loadModel()
            embMatrix = np.zeros((len(word_index) + 1, EMBEDDING_word2vec))
            for word, i in word_index.items():
                if word in embModel:
                    embVector = embModel[word]
                    embMatrix[i] = embVector
        else:
            tk = Tokenizer(num_words=vocabSize, char_level=char)
            tk.fit_on_texts(contents)
            tweetSequences = tk.texts_to_sequences(contents)
            tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
            #tweetVector = np.reshape(tweetVector, (tweetVector.shape[0], tweetVector.shape[1], 1))
    #tweet_train, tweet_test, time_train, time_test, label_train, label_test = train_test_split(tweetVector, timeVector, labels, test_size=0.2, random_state=42)

    # training
    print('training...')
    resultFile = open('result/result', 'a')
    accuracy = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(timeVector, activityLabels)):
        model_text = Sequential()
        if embedding == 'word2vec':
            #model_text.add(Masking(mask_value=0., input_shape=(None, EMBEDDING_DIM)))
            #model_text.add(InputLayer(input_shape=(None, None, EMBEDDING_DIM), name='embedding_input'))
            model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, EMBEDDING_word2vec)))
        elif embedding == 'tweet2vec':
            model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, EMBEDDING_tweet2vec)))
        elif embedding == 'gensim':
            model_text.add(Embedding(len(word_index)+1, EMBEDDING_word2vec, weights=[embMatrix], trainable=False))
            model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        else:
            model_text.add(Embedding(vocabSize, embeddingVectorLength, name='inputEmb'))
            #print model_text.output_shape
            model_text.add(Dropout(0.2))
            model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
            #model_text.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5)))


        model_time = Sequential()
        model_time.add(Dense(2, input_shape=(2,), activation='relu', name='time_input'))
        # merge text and time branches
        model = Sequential()
        model.add(Merge([model_text, model_time], mode='concat'))
        #model.add(Dropout(0.5))
        model.add(Dense(labelNum, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        tweet_train = tweetVector[train_index]
        time_train = timeVector[train_index]
        label_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        time_test = timeVector[test_index]
        label_test = labels[test_index]
        activityLabels_train = activityLabels[train_index]
        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', label_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, sample_weight=sampleWeight)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(activityLabels_train), activityLabels_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(countSum/labelCount[label])
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        elif balancedWeight == 'class_label_log':
            classWeight = []
            countSum = sum(labelCount.values())
            for label in labelList:
                classWeight.append(-math.log(labelCount[label]/countSum))
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        else:
            model.fit([tweet_train, time_train], label_train, validation_data=([tweet_test, time_test], label_test), epochs=3, batch_size=10, verbose=2)

        scores = model.evaluate([tweet_test, time_test],label_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        resultFile.write('Fold '+str(fold)+': '+str(scores[1] * 100)+'\n')
        accuracy += scores[1] * 100
    resultFile.write('Overall: '+str(accuracy/5)+'\n\n')
    print(accuracy/5)
    resultFile.close()


def processConv(labelNum, modelName, balancedWeight='None', embedding='None', clean=False, char=False):
    if embedding == 'tweet2vec':
        embData = np.load('embModel/embeddings.npy')
        tweetVector = np.reshape(embData, (embData.shape[0], 1, embData.shape[1]))
        print len(embData)
        labels = []
        labelFile = open('embModel/long1out.label', 'r')
        for line in labelFile:
            labels.append(line.strip())
        labelFile.close()
        activityLabels = np.array(labels)
        encoder = LabelEncoder()
        encoder.fit(labels)
        labelList = encoder.classes_.tolist()
        print labelList
        encodedLabels = encoder.transform(labels)
        labels = np_utils.to_categorical(encodedLabels)

        timeList = []
        timeFile = open('embModel/long1out.time', 'r')
        for line in timeFile:
            dateTemp = line.strip().split()
            timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
        timeFile.close()
        timeVector = np.array(timeList)
    else:
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

        contents = []
        labels = []
        timeList = []
        labelTweetCount = {}
        placeTweetCount = {}
        for index, place in enumerate(placeList):
            activity = activityList[index]
            if activity not in labelTweetCount:
                labelTweetCount[activity] = 0.0
            if clean:
                tweetFile = open('data/google_place_tweets3.3/POI_clean/' + place + '.json', 'r')
            else:
                tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
            tweetCount = 0
            for line in tweetFile:
                data = json.loads(line.strip())
                contents.append(data['text'].encode('utf-8'))
                dateTemp = data['created_at'].split()
                timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
                labels.append(activity)
                tweetCount += 1
            tweetFile.close()
            labelTweetCount[activity] += tweetCount
            placeTweetCount[place] = tweetCount
        activityLabels = np.array(labels)
        timeVector = np.array(timeList)
        encoder = LabelEncoder()
        encoder.fit(labels)
        labelList = encoder.classes_.tolist()
        print labelList
        encodedLabels = encoder.transform(labels)
        labels = np_utils.to_categorical(encodedLabels)

        if embedding == 'word2vec':
            w2v = word2vecReader.Word2Vec()
            embModel = w2v.loadModel()
            tweetVector, invalidList = contents2vecs(embModel, contents)
            timeVector = np.delete(timeVector, invalidList, 0)
            labels = np.delete(labels, invalidList, 0)
            tweetVector = np.reshape(tweetVector, (tweetVector.shape[0], 1, tweetVector.shape[1]))
        elif embedding == 'gensim':
            tk = Tokenizer(num_words=vocabSize, char_level=char)
            tk.fit_on_texts(contents)
            tweetSequences = tk.texts_to_sequences(contents)
            tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
            word_index = tk.word_index

            w2v = word2vecReader.Word2Vec()
            embModel = w2v.loadModel()
            embMatrix = np.zeros((len(word_index) + 1, EMBEDDING_word2vec))
            for word, i in word_index.items():
                if word in embModel:
                    embVector = embModel[word]
                    embMatrix[i] = embVector
        else:
            tk = Tokenizer(num_words=vocabSize, char_level=char)
            tk.fit_on_texts(contents)
            tweetSequences = tk.texts_to_sequences(contents)
            print tweetSequences[0]
            tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)

            print tweetVector.shape
            print tweetVector[0]
    # training
    print('training...')
    resultFile = open('result/result', 'a')
    accuracy = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(timeVector, activityLabels)):
        model_text = Sequential()
        if embedding == 'word2vec':
            # model_text.add(Masking(mask_value=0., input_shape=(None, EMBEDDING_DIM)))
            # model_text.add(InputLayer(input_shape=(None, None, EMBEDDING_DIM), name='embedding_input'))
            model_text.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, EMBEDDING_word2vec)))
        elif embedding == 'tweet2vec':
            model_text.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu', input_shape=(None, EMBEDDING_tweet2vec)))
        elif embedding == 'gensim':
            model_text.add(Embedding(len(word_index)+1, EMBEDDING_word2vec, input_length=tweetLength, weights=[embMatrix], trainable=False))
            model_text.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        else:
            #model_text.add(Embedding(vocabSize, embeddingVectorLength, input_length=tweetLength))
            #model_text.add(Dropout(0.2))
            model_text.add(InputLayer(input_shape=(None, vocabSize)))
            model_text.add(Conv1D(filters=64, kernel_size=10, padding='same', activation='relu'))
        # model_text.add(Dropout(0.2))
        #model_text.add(GlobalMaxPooling1D())
        #model_text.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        model_text.add(MaxPooling1D(pool_size=3))
        #model_text.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        #model_text.add(MaxPooling1D(pool_size=35))
        model_text.add(Flatten())
        #model_text.add(Dense(100, activation='relu'))

        model_time = Sequential()
        model_time.add(Dense(2, input_shape=(2,), activation='relu', name='time_input'))
        # merge text and time branches
        model = Sequential()
        model.add(Merge([model_text, model_time], mode='concat'))
        # model.add(Dropout(0.5))
        model.add(Dense(labelNum, activation='softmax', name='output_layer'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(model.summary())

        tweet_train = tweetVector[train_index]
        time_train = timeVector[train_index]
        label_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        time_test = timeVector[test_index]
        label_test = labels[test_index]
        activityLabels_train = activityLabels[train_index]
        if balancedWeight == 'sample':
            sampleWeight = compute_sample_weight('balanced', label_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, sample_weight=sampleWeight)
        elif balancedWeight == 'class':
            classWeight = compute_class_weight('balanced', np.unique(activityLabels_train), activityLabels_train)
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, class_weight=classWeight)
        else:
            model.fit([tweet_train, time_train], label_train, epochs=3, batch_size=10, verbose=0)

        scores = model.evaluate([tweet_test, time_test], label_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        resultFile.write('Fold ' + str(fold) + ': ' + str(scores[1] * 100) + '\n')
        accuracy += scores[1] * 100
    resultFile.write('Overall: ' + str(accuracy / 5) + '\n\n')
    print(accuracy / 5)
    resultFile.close()


def trainFullModel(labelNum, modelName, balancedWeight='None', char=False):
    #labelNum -= (len(excludeList) - 1)
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
            if line.strip() in excludeList:
                activityList.append('NONE')
            else:
                activityList.append(line.strip())
    activityListFile.close()

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
            tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
            tweetCount = 0
            for line in tweetFile:
                data = json.loads(line.strip())
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
    labelFile = open('model/'+modelName + '_' + str(balancedWeight) + '.label', 'w')
    labelFile.write(str(encoder.classes_).replace('\n', ' ').replace("'", "")[1:-1])
    labelFile.close()
    encodedLabels = encoder.transform(labels)
    labels = np_utils.to_categorical(encodedLabels)
    labelList = encoder.classes_.tolist()
    print(labels.shape)

    tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    pickle.dump(tk, open('model/' + modelName + '_' + str(balancedWeight) + '.tk', 'wb'))
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)

    model_text = Sequential()
    model_text.add(Embedding(vocabSize, embeddingVectorLength))
    model_text.add(Dropout(0.2))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model_text.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # model.add(Dropout(0.1))

    model_time = Sequential()
    model_time.add(Dense(2, input_shape=(2,), activation='relu'))
    # merge text and time branches
    model = Sequential()
    model.add(Merge([model_text, model_time], mode='concat'))
    model.add(Dense(labelNum, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    if balancedWeight == 'sample':
        sampleWeight = compute_sample_weight('balanced', labels)
        model.fit([tweetVector, timeVector], labels, epochs=3, batch_size=10, sample_weight=sampleWeight)
    elif balancedWeight == 'class':
        classWeight = compute_class_weight('balanced', np.unique(activityLabels), activityLabels)
        print np.unique(activityLabels).tolist()
        print labelCount
        print classWeight
        model.fit([tweetVector, timeVector], labels, epochs=3, batch_size=10, class_weight=classWeight)
    elif balancedWeight == 'class_label':
        classWeight = []
        countSum = sum(labelCount.values())
        for label in labelList:
            classWeight.append(countSum/labelCount[label])
        print classWeight
        model.fit([tweetVector, timeVector], labels, epochs=3, batch_size=10, class_weight=classWeight)
    elif balancedWeight == 'class_label_log':
        classWeight = []
        countSum = sum(labelCount.values())
        for label in labelList:
            classWeight.append(-math.log(labelCount[label] / countSum))
        print classWeight
        model.fit([tweetVector, timeVector], labels, epochs=3, batch_size=10, class_weight=classWeight)
    else:
        model.fit([tweetVector, timeVector], labels, epochs=3, batch_size=10)

    model_json = model.to_json()
    with open('model/'+modelName + '_' + str(balancedWeight) + '.json', 'w') as modelFile:
        modelFile.write(model_json)
    model.save_weights('model/' + modelName + '_' + str(balancedWeight) + '.h5')


if __name__ == "__main__":
    embedding = 'normal'

    #processAttLSTM(10, 'long1', 'class', clean=False, char=False)
    #processAttLSTM(6, 'long2', 'class', clean=False, char=False)
    processLSTM(10, 'long1', 'none', embedding, clean=False, char=False)
    #processCNNLSTM(3, 'long3', 'none', True)

    #processConv(10, 'long1', 'None')
    #processConv(6, 'long2', 'None')
    #processConv(3, 'long3', 'None')

    #trainFullModel(10, 'long1', 'class_label_log')
    #trainFullModel(6, 'long4', 'none')
    #trainFullModel(3, 'long5', 'none')
