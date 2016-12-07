import json
import statistics as st
import traceback

def userTweetConsolidator():
    brandList = []
    userSet = set()
    listFile = open('lists/popularAccount.list', 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    resultFile = open('results/brandUser.stat', 'w')
    for brand in brandList:
        print brand
        geoTweetCounts = []
        resultFile.write(brand+'\n')
        userTweetCounts = []
        userData = {}
        followerFile = open('data/followers/' + brand + '.json', 'r')
        for line in followerFile:
            userID = json.loads(line)['id']
            userSet.add(userID)
            userData[userID] = {}
        followerFile.close()

        inputFile = open('data/userTweets/'+brand+'.json', 'r')
        for line in inputFile:
            try:
                temp = json.loads(line.strip())
            except:
                traceback.print_exc()
                print line.strip()
            userID = temp['user']
            if userID not in userData:
                print userID
            else:
                tweets = temp['statuses']
                for tweet in tweets:
                    if len(tweet) == 7:
                        tweetID = tweet['id']
                        del tweet['id']
                        userData[userID][tweetID] = tweet

        inputFile.close()

        for userTweets in userData.values():
            userTweetCounts.append(len(userTweets))
            geoTweetCount = 0
            for tweet in userTweets.values():
                if tweet['coordinates'] is not None:
                    geoTweetCount += 1
            geoTweetCounts.append(geoTweetCount)

        outputFile = open('consolidated data/userTweet/'+brand+'.json', 'w')
        outputFile.write(json.dumps(userData))

        print sum(userTweetCounts)
        resultFile.write(str(sum(userTweetCounts))+'\n')
        print sum(geoTweetCounts)
        resultFile.write(str(sum(geoTweetCounts))+'\n')
        print str(st.mean(userTweetCounts)) + '\t' + str(st.variance(userTweetCounts))
        resultFile.write(str(st.mean(userTweetCounts)) + '\t' + str(st.variance(userTweetCounts)) + '\n')

    resultFile.close()

if __name__ == '__main__':
    userTweetConsolidator()