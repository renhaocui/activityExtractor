import json
import numpy as np
from keras.layers import Dense, Merge, MaxPooling1D, Conv1D, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from utilities import word2vecReader
from sklearn.metrics import confusion_matrix

vocabSize = 10000
tweetLength = 15
embeddingVectorLength = 200
EMBEDDING_word2vec = 400
EMBEDDING_tweet2vec = 500
charLengthLimit = 20

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


def processConv(modelName, balancedWeight='None', embedding='None', char=False, confMatrix=False):
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
    for index, place in enumerate(placeList):
        activity = activityList[index]
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

    tk = Tokenizer(num_words=vocabSize, char_level=char)
    tk.fit_on_texts(contents)
    tweetSequences = tk.texts_to_sequences(contents)
    tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')

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

        embMatrix = np.zeros((len(word_index)+1, 400))
        for word, i in word_index.items():
            if word in embModel:
                embMatrix[i] = embModel[word]

    #weightMatrix = np.ones((vocabSize, 1))

    # training
    print('training...')
    resultFile = open('result/result', 'a')
    accuracy = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(timeVector, activityLabels)):
        model_text = Sequential()
        if embedding == 'word2vec':
            model_text.add(
                Embedding(len(word_index)+1, 400, weights=[embMatrix], trainable=False, input_length=tweetLength))
        elif embedding == 'glove':
            model_text.add(
                Embedding(len(word_index) + 1, 100, weights=[embMatrix], trainable=False, input_length=tweetLength))
        else:
            model_text.add(Embedding(vocabSize, embeddingVectorLength, input_length=tweetLength))
        #model_text.add(Dropout(0.2))
        #model_text.add(InputLayer(input_shape=(None, vocabSize)))
        model_text.add(Conv1D(filters=64, kernel_size=10, padding='same', activation='relu'))
        # model_text.add(Dropout(0.2))
        #model_text.add(GlobalMaxPooling1D())
        #model_text.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        print model_text.output_shape
        model_text.add(MaxPooling1D(pool_size=3))
        #model_text.add(Conv1D(filters=64, kernel_size=5, padding='same', activation='relu'))
        #model_text.add(MaxPooling1D(pool_size=35))
        print model_text.output_shape
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
        activityLabels_test = activityLabels[test_index]
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

        if confMatrix:
            predictions = model.predict([tweet_test, time_test])
            # sampleFile = open('result/sample', 'a')
            predLabels = []
            for index, pred in enumerate(predictions):
                if index % 100 == 0:
                    predLabel = labelList[pred.tolist().index(max(pred))]
                    # sampleFile.write(contents_test[index]+'\t'+activityLabels_test[index]+'\t'+predLabel+'\n')
                predLabels.append(predLabel)
            confusionFile = open('result/' + modelName + '_' + balancedWeight + '.confMatrix', 'a')
            conMatrix = confusion_matrix(activityLabels_test, predLabels)
            for row in conMatrix:
                lineOut = ''
                for line in row:
                    lineOut += str(line) + '\t'
                confusionFile.write(lineOut.strip() + '\n')
            confusionFile.write('\n')
            confusionFile.close()
            # sampleFile.close()

    resultFile.write('Overall: ' + str(accuracy / 5) + '\n\n')
    print(accuracy / 5)
    resultFile.close()



if __name__ == "__main__":
    embedding = 'normal'

    processConv('long0', 'none', embedding, char=False, confMatrix=True)
    processConv('long0', 'class', embedding, char=False, confMatrix=False)