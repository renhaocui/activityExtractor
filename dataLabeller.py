# -*- coding: utf-8 -*-
import json
import tweetTextCleaner
import random
import utilities


def label():
    inputFile = open('data/experiment/data.json', 'r')
    parseKeyFile = open('data/experiment/tweet.parse.key', 'r')
    for line in parseKeyFile:
        temp = line.strip().split(',')



def googlePlaceTweetlabeller(folder):
    placeList = []
    placeActivityMapper = {}
    placeListFile = open('lists/google_place.category', 'r')
    for line in placeListFile:
        placeList.append(line.strip())
    placeListFile.close()
    activityListFile = open('lists/google_place_activity.list', 'r')
    for index, line in enumerate(activityListFile):
        placeActivityMapper[placeList[index]] = line.strip()
    activityListFile.close()

    contentFile = open('data/experiment/tweet.content', 'w')
    labelFile = open('data/experiment/tweet.label', 'w')
    placeFile = open('data/experiment/tweet.place', 'w')
    sampleTextFile = open('data/sample/googleTweet.content', 'w')
    sampleLabelFile = open('data/sample/googleTweet.label', 'w')
    samplePlaceFile = open('data/sample/googleTweet.place', 'w')
    data = {}
    index = 0
    for place in placeList:
        inputFile = open(folder + place + '.json', 'r')
        for line in inputFile:
            index += 1
            tweet = json.loads(line.strip())
            id = tweet['id']
            tweet.pop('id', None)
            tweet['text'] = tweet['text'].replace('\n', ' ').replace('\r', ' ')
            tempList = []
            for place in tweet['google_place_category']:
                if place in placeActivityMapper:
                    tempList.append(placeActivityMapper[place])
            tweet['activities'] = tempList
            data[id] = tweet
            contentFile.write(tweet['text'].encode('utf-8') + '\n')
            labelFile.write(utilities.listToStr(tweet['activities']) + '\n')
            placeFile.write(utilities.listToStr(tweet['google_place_category']) + '\n')
            if index % 200 == 0:
                sampleTextFile.write(tweet['text'].encode('utf-8') + '\n')
                sampleLabelFile.write(utilities.listToStr(tweet['activities']) + '\n')
                samplePlaceFile.write(utilities.listToStr(tweet['google_place_category']) + '\n')

    sampleTextFile.close()
    sampleLabelFile.close()
    samplePlaceFile.close()
    contentFile.close()
    labelFile.close()
    placeFile.close()

    dataFile = open('data/experiment/data.json', 'w')
    dataFile.write(json.dumps(data))
    dataFile.close()
    print len(data)
    '''
    NERList = utilities.NERExtractor('data/labelled_data/googleTweet.content.ner')
    nerFile = open('data/labelled_data/googleTweet.ner', 'w')
    for item in NERList:
        nerFile.write(item+'\n')
    nerFile.close()
    '''


def labeller():
    placeList = []
    activityList = []
    placeListFile = open('lists/google_place.category', 'r')
    for line in placeListFile:
        placeList.append(line.strip())
    placeListFile.close()
    activityListFile = open('lists/google_place_activity.list', 'r')
    for line in activityListFile:
        activityList.append(line.strip())
    activityListFile.close()

    placeActivityMapper = {}
    for i in range(len(placeList)):
        placeActivityMapper[placeList[i]] = activityList[i]

    inputFile = open('data/place_labelled_tweet/poi_tweet.json', 'r')
    outputFile = open('data/labelled_data/googleTweet.json', 'w')
    contentFile = open('data/labelled_data/googleTweet.content', 'w')
    categoryFile = open('data/labelled_data/googleTweet.category', 'w')
    labelFile = open('data/labelled_data/googleTweet.label', 'w')
    labelsFile = open('data/labelled_data/googleTweet.labels', 'w')
    nameFile = open('data/labelled_data/googleTweet.name', 'w')
    timeFile = open('data/labelled_data/googleTweet.time', 'w')
    for index, line in enumerate(inputFile):
        item = json.loads(line.strip())
        content = tweetTextCleaner.tweetCleaner(item['text']).encode('utf-8')
        # content = item['text'].encode('utf-8')
        contentFile.write(content + '\n')
        categories = item['google_place_category']
        name = item['twitter_place_name'].encode('utf-8')
        labels = ''
        outputLabels = []
        for cate in categories:
            if cate in placeActivityMapper:
                outputLabels.append(placeActivityMapper[cate])
                labels += placeActivityMapper[cate] + ' '
        labelsFile.write(labels.strip() + '\n')
        label = item['collect_place_category']
        categoryFile.write(label + '\n')
        timeFile.write(item['created_at'] + '\n')
        nameFile.write(name + '\n')
        if label in placeActivityMapper:
            item['label'] = placeActivityMapper[label]
            labelFile.write(placeActivityMapper[label] + '\n')
        else:
            item['label'] = None
            labelFile.write('\n')
        item.pop('google_palce_category', None)
        item.pop('collect_place_category', None)
        item['labels'] = outputLabels
        outputFile.write(json.dumps(item) + '\n')

    nameFile.close()
    inputFile.close()
    outputFile.close()
    contentFile.close()
    labelFile.close()
    labelsFile.close()
    categoryFile.close()
    timeFile.close()

    NERList = utilities.NERExtractor('data/labelled_data/googleTweet.content.ner')
    nerFile = open('data/labelled_data/googleTweet.ner', 'w')
    for item in NERList:
        nerFile.write(item + '\n')
    nerFile.close()

    print index


def sampler():
    contentFile = open('data/labelled_data/googleTweet.content', 'r')
    labelFile = open('data/labelled_data/googleTweet.label', 'r')
    labelsFile = open('data/labelled_data/googleTweet.labels', 'r')
    nameFile = open('data/labelled_data/googleTweet.name', 'r')
    categoryFile = open('data/labelled_data/googleTweet.category', 'r')
    timeFile = open('data/labelled_data/googleTweet.time', 'r')
    nerFile = open('data/labelled_data/googleTweet.ner', 'r')

    content = []
    label = []
    labels = []
    name = []
    time = []
    category = []
    ner = []

    for index, line in enumerate(contentFile):
        content.append(line)
    totalNum = index
    for line in labelFile:
        label.append(line)
    for line in labelsFile:
        labels.append(line)
    for line in nameFile:
        name.append(line)
    for line in categoryFile:
        category.append(line)
    for line in timeFile:
        time.append(line)
    for line in nerFile:
        ner.append(line)

    contentFile.close()
    labelFile.close()
    labelsFile.close()
    nameFile.close()
    categoryFile.close()
    timeFile.close()
    nerFile.close()
    indexSet = []
    while (True):
        if len(indexSet) > 100:
            break
        num = random.randint(0, totalNum)
        if num not in indexSet:
            indexSet.append(num)

    sampleContentFile = open('data/labelled_data/googleTweetSample.content', 'w')
    sampleLabelFile = open('data/labelled_data/googleTweetSample.label', 'w')
    sampleLabelsFile = open('data/labelled_data/googleTweetSample.labels', 'w')
    sampleNameFile = open('data/labelled_data/googleTweetSample.name', 'w')
    sampleCategoryFile = open('data/labelled_data/googleTweetSample.category', 'w')
    sampleTimeFile = open('data/labelled_data/googleTweetSample.time', 'w')
    sampleNerFile = open('data/labelled_data/googleTweetSample.ner', 'w')

    for index in indexSet:
        sampleContentFile.write(content[index])
        sampleLabelFile.write(label[index])
        sampleLabelsFile.write(labels[index])
        sampleNameFile.write(name[index])
        sampleCategoryFile.write(category[index])
        sampleTimeFile.write(time[index])
        sampleNerFile.write(ner[index])

    sampleContentFile.close()
    sampleLabelFile.close()
    sampleLabelsFile.close()
    sampleNameFile.close()
    sampleCategoryFile.close()
    sampleTimeFile.close()
    sampleNerFile.close()
    print totalNum


if __name__ == "__main__":
    googlePlaceTweetlabeller('data/google_place_tweets3.3/')
    # sampler()
