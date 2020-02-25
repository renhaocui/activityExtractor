import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import *
from scipy.sparse import hstack, csr_matrix
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

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

def process(modelName, confMatrix=False):
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
            tweetFile = open('data/POIplace/' + place + '.json', 'r')
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
    timeVector = np.array(timeList)
    labels = np.array(labels)
    labelFile = open('result/tradition.label', 'a')
    labelFile.write(str(np.unique(labels)) + '\n')
    labelFile.close()

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 5), min_df=2, stop_words='english',
                                 binary='True')
    vectorMatrix = vectorizer.fit_transform(contents)
    tweetVector = hstack((vectorMatrix, csr_matrix(timeVector)), format='csr')

    # training
    print('training...')
    resultFile = open('result/result', 'a')
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    contents = np.array(contents)
    for fold, (train_index, test_index) in enumerate(skf.split(tweetVector, labels)):
        print fold
        tweet_train = tweetVector[train_index]
        label_train = labels[train_index]
        tweet_test = tweetVector[test_index]
        label_test = labels[test_index]
        contents_test = contents[test_index]

        model = svm.SVC(probability=True)
        model.fit(tweet_train, label_train)

        predictions = model.predict(tweet_test)

        accuracy += accuracy_score(label_test, predictions)
        precision += precision_score(label_test, predictions, average='micro')
        recall += recall_score(label_test, predictions, average='micro')
        f1 += f1_score(label_test, predictions, average='micro')

        sampleFile = open('result/tradition.sample', 'a')
        for index, pred in enumerate(predictions):
            sampleFile.write(contents_test[index] + '\t' + label_test[index] + '\t' + pred + '\n')
        sampleFile.close()
        if confMatrix:
            confusionFile = open('result/tradition.confMatrix', 'a')
            conMatrix = confusion_matrix(label_test, predictions)
            for row in conMatrix:
                lineOut = ''
                for line in row:
                    lineOut += str(line) + '\t'
                confusionFile.write(lineOut.strip() + '\n')
            confusionFile.write('\n')
            confusionFile.close()

    resultFile.write('Accuracy: ' + str(accuracy / 5) + '\n\n')
    resultFile.write('Precision: ' + str(precision / 5) + '\n\n')
    resultFile.write('Recall: ' + str(recall / 5) + '\n\n')
    resultFile.write('F1: ' + str(recall / 5) + '\n\n')
    print(accuracy*100 / 5)
    print(precision*100 / 5)
    print(recall*100 / 5)
    print(f1*100 / 5)
    resultFile.close()



if __name__ == '__main__':
    process('long1', confMatrix=True)