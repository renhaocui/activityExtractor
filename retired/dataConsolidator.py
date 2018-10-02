import json
import statistics as st
import traceback
import os
from datetime import datetime, date

def userTweetConsolidator():
    brandList = []
    listFile = open('lists/popularAccount.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    totalUser = 0.0
    totalActiveUser = 0.0
    totalTweet = 0.0
    totalActiveTweet = 0.0

    resultFile = open('results/brandUser.stat', 'w')
    for brand in brandList:
        print brand
        geoTweetCounts = []
        resultFile.write(brand+'\n')
        averageTweetCounts = []
        activeUserCounts = 0.0
        userTweetCounts = []
        activeUserTweetCounts = []
        userData = {}

        inputFile = open('data/userTweets/'+brand+'.json', 'r')
        for line in inputFile:
            try:
                temp = json.loads(line.strip())
            except:
                traceback.print_exc()
                print line.strip()
            userID = temp['user']
            if len(temp['statuses']) > 0:
                if userID not in userData:
                    userData[userID] = {}
                else:
                    tweets = temp['statuses']
                    for tweet in tweets:
                        if len(tweet) == 7:
                            try:
                                tweetID = tweet['id']
                                del tweet['id']
                                tempTime = tweet['created_at'].split()
                                createdDate = tempTime[1]+' '+tempTime[2]+' '+tempTime[5]
                                dateObject = datetime.strptime(createdDate, '%b %d %Y')
                                days = datetime.today() - dateObject
                                tweet['range'] = float(str(days).split()[0])
                                userData[userID][tweetID] = tweet
                            except:
                                traceback.print_exc()
                                print tweet
        inputFile.close()

        outputData = {}
        for userID, userTweets in userData.items():
            if len(userTweets) > 0:
                outputData[userID] = userTweets
                userTweetCounts.append(len(userTweets))
                daysList = []
                geoTweetCount = 0.0
                for tweet in userTweets.values():
                    daysList.append(tweet['range'])
                    if tweet['coordinates'] is not None:
                        geoTweetCount += 1
                averageTweetCounts.append(len(userTweets)/max(daysList))
                if len(userTweets) > max(daysList):
                    activeUserCounts += 1
                    activeUserTweetCounts.append(len(userTweets))
                geoTweetCounts.append(geoTweetCount)

        outputFile = open('consolidated data/userTweet/'+brand+'.json', 'w')
        outputFile.write(json.dumps(outputData))
        outputFile.close()

        totalUser += len(userTweetCounts)
        totalActiveUser += activeUserCounts
        totalTweet += sum(userTweetCounts)
        totalActiveTweet += sum(activeUserTweetCounts)

        print len(userTweetCounts)
        resultFile.write(str(len(userTweetCounts))+'\n')
        print activeUserCounts
        resultFile.write(str(activeUserCounts)+'\n')
        print sum(userTweetCounts)
        resultFile.write(str(sum(userTweetCounts))+'\n')
        print st.mean(averageTweetCounts)
        resultFile.write(str(st.mean(averageTweetCounts))+'\n')
        print str(sum(geoTweetCounts))
        resultFile.write(str(sum(geoTweetCounts)))
        print str(st.mean(userTweetCounts))
        resultFile.write(str(st.mean(userTweetCounts)))
    resultFile.close()

    print 'Total user: '+str(totalUser)
    print 'Total active user: '+str(totalActiveUser)
    print 'Average tweets: '+str(totalTweet/totalUser)
    print 'Average active tweets: '+str(totalActiveTweet/totalActiveUser)


def placeTweetConsolidator():
    totalPlaces = 0.0
    totalTweets = 0.0
    totalActivePlaces = 0.0
    totalActiveTweets = 0.0

    resultFile = open('results/placeTweet.stat', 'w')
    for file in os.listdir("data/google_place_tweets3.3"):
        place = file.split('.')[0]
        print place
        activePlaceCount = 0.0
        activePlaceTweetCount = []
        resultFile.write(place+'\n')
        inputFile = open('data/google_place_tweets3.3/' + file, 'r')
        placeData = {}
        for line in inputFile:
            try:
                data = json.loads(line.strip())
                googlePlaceName =data['google_place_name']
                if googlePlaceName not in placeData:
                    placeData[googlePlaceName] = {}
                tweetID = data['id']
                del data['id']
                tempTime = data['created_at'].split()
                createdDate = tempTime[1] + ' ' + tempTime[2] + ' ' + tempTime[5]
                dateObject = datetime.strptime(createdDate, '%b %d %Y')
                days = datetime.today() - dateObject
                data['range'] = float(str(days).split()[0])
                placeData[googlePlaceName][tweetID] = data
            except:
                traceback.print_exc()
                print tweet
        inputFile.close()

        outputFile = open('consolidated data/placeTweet/'+place+'.json', 'w')
        outputFile.write(json.dumps(placeData))
        outputFile.close()

        outputContentFile = open('consolidated data/placeTweet/'+place+'.content', 'w')
        placeTweetCount = []
        for tweets in placeData.values():
            daysList = []
            placeTweetCount.append(len(tweets))
            for tweet in tweets.values():
                daysList.append(tweet['range'])
                outputContentFile.write(tweet['text'].encode('utf-8').replace('\n', ' ').replace('\r', ' ')+'\n')

            if len(placeTweetCount) > max(daysList):
                activePlaceCount += 1
                activePlaceTweetCount.append(len(placeTweetCount))
        outputContentFile.close()

        totalPlaces += len(placeTweetCount)
        totalTweets += sum(placeTweetCount)
        totalActivePlaces += activePlaceCount
        totalActiveTweets += sum(activePlaceTweetCount)

        print len(placeTweetCount)
        resultFile.write(str(len(placeTweetCount))+'\n')
        print sum(placeTweetCount)
        resultFile.write(str(sum(placeTweetCount))+'\n')
        print str(st.mean(placeTweetCount))+'\t'+str(st.variance(placeTweetCount))
        resultFile.write(str(st.mean(placeTweetCount))+'\t'+str(st.variance(placeTweetCount))+'\n')

    resultFile.close()

    print 'Total places: '+str(totalPlaces)
    print 'Total active places: '+str(totalActivePlaces)
    print 'Average tweets: '+str(totalTweets/totalPlaces)
    print 'Average active tweets: '+str(totalActiveTweets/totalActivePlaces)

if __name__ == '__main__':
    #userTweetConsolidator()
    placeTweetConsolidator()