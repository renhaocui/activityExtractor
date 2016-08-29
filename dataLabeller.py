# -*- coding: utf-8 -*-
import json
import tweetTextCleaner

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
    labelFile = open('data/labelled_data/googleTweet.label', 'w')
    labelsFile = open('data/labelled_data/googleTweet.labels', 'w')
    nameFile = open('data/labelled_data/googleTweet.name', 'w')
    for index, line in enumerate(inputFile):
        item = json.loads(line.strip())
        content = tweetTextCleaner.tweetCleaner(item['text']).encode('utf-8')
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

    print index

if __name__ == "__main__":
    labeller()