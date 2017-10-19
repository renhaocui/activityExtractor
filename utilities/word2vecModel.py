import json

import numpy
import sklearn.metrics
from sklearn import linear_model
from sklearn import svm
from sklearn.neural_network import MLPClassifier

import word2vecReader
from tokenizer import simpleTokenize


def scoreToBinary(inputList, splitNum):
    outputList = []
    for item in inputList:
        if item > splitNum:
            outputList.append(1)
        else:
            outputList.append(0)

    return outputList

def evaluate(predictions, test_labels, mode, splitNum):
    if len(predictions) != len(test_labels):
        print 'prediction error!'
        return 404
    if mode == 1:
        test_labels = scoreToBinary(test_labels, splitNum)
        predictions = scoreToBinary(predictions, splitNum)
        precision = sklearn.metrics.precision_score(test_labels, predictions)
        recall = sklearn.metrics.recall_score(test_labels, predictions)
        F1 = sklearn.metrics.f1_score(test_labels, predictions)
        auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc
    else:
        precision = sklearn.metrics.precision_score(test_labels, predictions)
        recall = sklearn.metrics.recall_score(test_labels, predictions)
        F1 = sklearn.metrics.f1_score(test_labels, predictions)
        auc = sklearn.metrics.roc_auc_score(test_labels, predictions)
        return precision, recall, F1, auc


def content2vec(model, content):
    words = simpleTokenize(content)
    tempList = []
    for word in words:
        if word in model.vocab:
            tempList.append(model[word])
    if len(tempList) < 1:
        return None
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
    return numpy.array(output)


def runModel(labelMode, splitNum, trainMode):
    w2v = word2vecReader.Word2Vec()
    model = w2v.loadModel()

    indexFile = open('dataset/experiment/groups/totalGroup2/indices/3.indices', 'r')
    for line in indexFile:
        temp = json.loads(line.strip())
        trainIDs = temp['train']
        testIDs = temp['test']
    indexFile.close()

    print 'building tweet vectors...'
    tweetData = {}
    dataFile = open('dataset/experiment/clean.labeled', 'r')
    for num, line in enumerate(dataFile):
        item = json.loads(line.strip())
        if labelMode == 1:
            label = item['label']
        else:
            if item['label'] > splitNum:
                label = 1
            else:
                label = 0
        content = item['content']
        tweetVec = content2vec(model, content)
        if tweetVec != None:
            tweetData[str(item['id'])] = {'vector': tweetVec, 'label': label}
    dataFile.close()
    print len(tweetData)

    trainFeature = []
    trainLabel = []
    for tweetID in trainIDs:
        if tweetID in tweetData:
            trainFeature.append(tweetData[tweetID]['vector'])
            trainLabel.append(tweetData[tweetID]['label'])

    testFeature = []
    testLabel = []
    for tweetID in testIDs:
        if tweetID in tweetData:
            testFeature.append(tweetData[tweetID]['vector'])
            testLabel.append(tweetData[tweetID]['label'])

    if trainMode == 'MaxEnt':
        model = linear_model.LogisticRegression()
    elif trainMode == 'SVM':
        model = svm.SVC()
    else:
        model = MLPClassifier(activation='logistic', learning_rate='constant')

    print 'Training...'
    model.fit(trainFeature, trainLabel)
    print 'Inference...'
    predictions = model.predict(testFeature)
    precision, recall, F1, auc = evaluate(predictions, testLabel, labelMode, splitNum)

    print precision
    print recall
    print F1
    print auc

if __name__ == "__main__":
    runModel(2, 5, 'SVM')