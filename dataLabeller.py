# -*- coding: utf-8 -*-
import json
import tweetTextCleaner
import random

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
        #content = item['text'].encode('utf-8')
        contentFile.write(content+'\n')
        categories = item['google_place_category']
        name = item['twitter_place_name'].encode('utf-8')
        labels = ''
        outputLabels = []
        for cate in categories:
            if cate in placeActivityMapper:
                outputLabels.append(placeActivityMapper[cate])
                labels += placeActivityMapper[cate] + ' '
        labelsFile.write(labels.strip()+'\n')
        label = item['collect_place_category']
        categoryFile.write(label+'\n')
        timeFile.write(item['created_at']+'\n')
        nameFile.write(name+'\n')
        if label in placeActivityMapper:
            item['label'] = placeActivityMapper[label]
            labelFile.write(placeActivityMapper[label]+'\n')
        else:
            item['label'] = None
            labelFile.write('\n')
        item.pop('google_palce_category', None)
        item.pop('collect_place_category', None)
        item['labels'] = outputLabels
        outputFile.write(json.dumps(item)+'\n')

    nameFile.close()
    inputFile.close()
    outputFile.close()
    contentFile.close()
    labelFile.close()
    labelsFile.close()
    categoryFile.close()
    timeFile.close()
    print index


def sampler():
    contentFile = open('data/labelled_data/googleTweet.content', 'r')
    labelFile = open('data/labelled_data/googleTweet.label', 'r')
    labelsFile = open('data/labelled_data/googleTweet.labels', 'r')
    nameFile = open('data/labelled_data/googleTweet.name', 'r')
    categoryFile = open('data/labelled_data/googleTweet.category', 'r')
    timeFile = open('data/labelled_data/googleTweet.time', 'r')
    content = []
    label = []
    labels = []
    name = []
    time = []
    category = []
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

    contentFile.close()
    labelFile.close()
    labelsFile.close()
    nameFile.close()
    categoryFile.close()
    timeFile.close()
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

    for index in indexSet:
        sampleContentFile.write(content[index])
        sampleLabelFile.write(label[index])
        sampleLabelsFile.write(labels[index])
        sampleNameFile.write(name[index])
        sampleCategoryFile.write(category[index])
        sampleTimeFile.write(time[index])

    sampleContentFile.close()
    sampleLabelFile.close()
    sampleLabelsFile.close()
    sampleNameFile.close()
    sampleCategoryFile.close()
    timeFile.close()

    print totalNum

if __name__ == "__main__":
    labeller()
    sampler()