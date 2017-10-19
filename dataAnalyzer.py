import json


def statGenerator(modelName, lengthLimit):
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

    labelTweetCount = {}
    placeTweetCount = {}
    for index, place in enumerate(placeList):
        activity = activityList[index]
        if activity not in labelTweetCount:
            labelTweetCount[activity] = 0.0
        if place not in placeTweetCount:
            placeTweetCount[place] = 0.0
        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        tweetCount = 0
        for line in tweetFile:
            data = json.loads(line.strip())
            if len(data['text']) > lengthLimit:
                tweetCount += 1
        tweetFile.close()
        labelTweetCount[activity] += tweetCount
        placeTweetCount[place] += tweetCount

    #for place, count in placeTweetCount.items():
    #    print place + '\t' + str(count)
    #print '\n\n'
    for label, count in labelTweetCount.items():
        print label + '\t' + str(count)

if __name__ == '__main__':
    statGenerator('long1.5', 20)