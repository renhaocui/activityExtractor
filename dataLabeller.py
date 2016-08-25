import json
import os

def labeller():
    data = {}
    idSet = set()
    categorySet = set()
    outputTypeFile = open('data/twitter_place_category.list', 'w')
    for file in os.listdir("google_place_tweets"):
        inputFile = open('google_place_tweets/'+file, 'r')
        google_category = file.split('.')[0]
        for line in inputFile:
            item = json.loads(line.strip())
            if item['id'] not in idSet:
                idSet.add(item['id'])
                data[item['id']] =
            else:



        inputFile.close()


    outputTypeFile.close()