import json
import os

def GoogleLabeller(inFileDir):
    idSet = set()
    placeType = {}
    googlePlaceCategory = {}
    outputFilePOI = open('data/place_labelled_tweet/poi_tweet.json', 'w')
    outputFileNB = open('data/place_labelled_tweet/neighborhood_tweet.json', 'w')
    for file in os.listdir(inFileDir):
        inputFile = open(inFileDir+'/'+file, 'r')
        google_category = file.split('.')[0]
        googlePlaceCategory[google_category] = 0
        for line in inputFile:
            item = json.loads(line.strip())
            if item['id'] not in idSet:
                idSet.add(item['id'])
                #item.pop('google_place_category', None)
                if 'place' in item:
                    if item['place'] is not None:
                        tweet_category = item['place']['place_type']
                        item['collect_place_category'] = google_category
                        if tweet_category == 'poi':
                            item['twitter_place_name'] = item['place']['full_name']
                            item.pop('place', None)
                            outputFilePOI.write(json.dumps(item)+'\n')
                            googlePlaceCategory[google_category] += 1
                        elif tweet_category == 'neighborhood':
                            outputFileNB.write(json.dumps(item) + '\n')
                        if tweet_category not in placeType:
                            placeType[tweet_category] = 1
                        else:
                            placeType[tweet_category] += 1

        inputFile.close()

    outputFileNB.close()
    outputFilePOI.close()

    print 'Google Place Category: '
    print googlePlaceCategory
    print 'Twitter Place Type: '
    print placeType

def dataMerger(file1, file2):
    inputFile1 = open(file1, 'r')
    inputFile2 = open(file2, 'r')
    idSet = set()
    outputFile = open('data/place_labelled_tweet/poi_tweet.json', 'w')
    for line in inputFile1:
        data = json.loads(line.strip())
        id = data['id']
        if id not in idSet:
            idSet.add(id)
            outputFile.write(line)
    inputFile1.close()
    for line in inputFile2:
        data = json.loads(line.strip())
        id = data['id']
        if id not in idSet:
            idSet.add(id)
            outputFile.write(line)
    inputFile2.close()

def TomTomLabeller():
    idSet = set()
    placeType = {}
    tomtomPlaceCategory = {}
    outputFilePOI = open('data/place_labelled_tweet/poi_tweet(tomtom).json', 'w')
    outputFileNB = open('data/place_labelled_tweet/neighborhood_tweet(tomtom).json', 'w')
    for file in os.listdir("data/tomtom_place_tweets"):
        inputFile = open('data/tomtom_place_tweets/'+file, 'r')
        tomtom_category = file.split('.')[0]
        tomtomPlaceCategory[tomtom_category] = 0
        for line in inputFile:
            item = json.loads(line.strip())
            if item['id'] not in idSet:
                idSet.add(item['id'])
                item.pop('retweet_count', None)
                if 'place' in item:
                    if item['place'] != None:
                        tweet_category = item['place']['place_type']
                        item['place_category'] = tomtom_category
                        if tweet_category == 'poi':
                            outputFilePOI.write(json.dumps(item)+'\n')
                            tomtomPlaceCategory[tomtom_category] += 1
                        elif tweet_category == 'neighborhood':
                            outputFileNB.write(json.dumps(item) + '\n')
                        if tweet_category not in placeType:
                            placeType[tweet_category] = 1
                        else:
                            placeType[tweet_category] += 1

        inputFile.close()

    outputFileNB.close()
    outputFilePOI.close()

    print 'TomTom Place Category: '
    print tomtomPlaceCategory
    print 'Twitter Place Type: '
    print placeType


if __name__ == "__main__":
    GoogleLabeller("data/google_place_tweets3.1")
    #TomTomLabeller()
    #dataMerger('data/place_labelled_tweet/poi_tweet(google).json', 'data/place_labelled_tweet/poi_tweet(google2).json')