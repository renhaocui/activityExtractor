import json
from sklearn.feature_extraction.text import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn import cluster
from scipy.sparse import csr_matrix, vstack


def merger(modelName, outGroupNum):
    embData = np.load('embModel/embeddings.npy')
    print type(embData)
    print len(embData)

    labels = []
    labelFile = open('embModel/long1out.label', 'r')
    for line in labelFile:
        labels.append(line.strip())
    labelFile.close()
    print len(labels)

    labelTweetData = {}
    for index, label in enumerate(labels):
        if label not in labelTweetData:
            labelTweetData[label] = []
        labelTweetData[label].append(embData[index])

    labelTweetVector = []
    labels = []
    for label in labelTweetData:
        labelTweetVector.append(np.mean(labelTweetData[label], axis=0))
        labels.append(label)

    print('Running Kmeans++...')
    kmeans = cluster.KMeans(n_clusters=outGroupNum, init='k-means++')
    kmeans.fit(labelTweetVector)
    resultFile = open('result/'+modelName+'_'+str(outGroupNum)+'.mapping', 'w')
    for index, label in enumerate(kmeans.labels_):
        resultFile.write(labels[index] + ' ' + str(label)+'\n')
    resultFile.close()


def merger3(labelNum, modelName, outGroupNum):
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
    for index, place in enumerate(placeList):
        activity = activityList[index]
        tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            contents.append(data['text'].encode('utf-8'))
            labels.append(activity)
        tweetFile.close()

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english', binary='True')
    #vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    tweet_matrix = vectorizer.fit_transform(contents)
    print(tweet_matrix.shape)
    tweetCount = tweet_matrix.shape[0]

    labelList = []
    vectorList = []
    labelVectorList = {}
    print('Generating center vectors for each label...')
    for index in range(tweetCount):
        label = labels[index]
        if label not in labelVectorList:
            labelVectorList[label] = tweet_matrix[index]
        else:
            labelVectorList[label] = vstack(labelVectorList[label], tweet_matrix[index])
    for label, matrix in labelVectorList.items():
        labelList.append(label)
        vectorList.append(csr_matrix.mean(matrix, axis=0))
    matrix = csr_matrix(np.array(vectorList))

    print('Running Kmeans++...')
    kmeans = cluster.KMeans(n_clusters=outGroupNum, init='k-means++')
    kmeans.fit(matrix)
    for index, label in enumerate(kmeans.labels_):
        print labelList[index] + ': ' + str(label)


def merger2(labelNum, modelName, outNum):
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

    for index, place in enumerate(placeList):
        activity = activityList[index]
        tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
        tweetCount = 0
        for line in tweetFile:
            data = json.loads(line.strip())
            contents.append(data['text'].encode('utf-8'))
            labels.append(activity)
        tweetFile.close()

    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1), min_df=1, stop_words='english', binary='True')
    #vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), min_df=0, stop_words='english')
    tweet_matrix = vectorizer.fit_transform(contents)
    print(tweet_matrix.shape)
    tweetCount = tweet_matrix.shape[0]
    vectorLength = tweet_matrix.shape[1]
    print('Calculating tweet pairwise similarity....')
    similarity = cosine_similarity(tweet_matrix, tweet_matrix)
    print(similarity.shape)

    print('Generating label pairwise similarity...')
    totalSimilarity = {}
    totalCount = {}
    avgSimilarity = {}
    for indexRow in range(tweetCount):
        labelRow = labels[indexRow]
        if labelRow not in totalSimilarity:
            totalSimilarity[labelRow] = {}
        if labelRow not in totalCount:
            totalCount[labelRow] = {}
        for indexCol in range(tweetCount):
            labelCol = labels[indexCol]
            if indexCol < indexRow:
                if labelCol not in totalSimilarity[labelRow]:
                    totalSimilarity[labelRow][labelCol] = 0.0
                if labelCol not in totalCount[labelRow]:
                    totalCount[labelRow][labelCol] = 0.0
                totalSimilarity[labelRow][labelCol] += similarity[indexRow][indexCol]
                totalCount[labelRow][labelCol] += 1.0
            else:
                if labelCol not in totalCount[labelRow]:
                    totalCount[labelRow][labelCol] = 0.0
                totalSimilarity[labelRow][labelCol] = 0
                totalCount[labelRow][labelCol] += 1.0

    for labelRow in activityList:
        if labelRow not in avgSimilarity:
            avgSimilarity[labelRow] = {}
            for labelCol in activityList:
                if labelCol not in avgSimilarity[labelRow]:
                    avgSimilarity[labelRow][labelCol] = totalSimilarity[labelRow][labelCol]/totalCount[labelRow][labelCol]
    print avgSimilarity



def modify(oldModelName, outGroupNum, newModelName):
    labelMapper = {}
    mappingFile = open('result/' + oldModelName + '_' + str(outGroupNum) + '.mapping', 'r')
    for line in mappingFile:
        items = line.strip().split()
        labelMapper[items[0]] = items[1]
    mappingFile.close()

    oldLabelFile = open('lists/google_place_activity_'+oldModelName+'.list', 'r')
    newLabelFile = open('lists/google_place_activity_'+newModelName+'.list', 'w')
    for line in oldLabelFile:
        newLabelFile.write(labelMapper[line.strip()]+'\n')
    oldLabelFile.close()
    newLabelFile.close()

if __name__ == "__main__":
    #merger('long1', 3)
    #merger('long1', 6)
    modify('long1', 3, 'long5')