from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Merge
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import math, pickle, json

vocabSize = 10000
tweetLength = 15
embeddingVectorLength = 200
charLengthLimit = 20

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}

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


if __name__ == '__main__':
    #trainLSTM('long0', 'none', char=False)
    #trainLSTM('long0', 'class', char=False)
    #trainLSTM('long0', 'sample', char=False)
    trainLSTM('long1.5', 'none', char=False)
    #trainLSTM('long1.0', 'class', char=False)
    #trainLSTM('long1.0', 'sample', char=False)