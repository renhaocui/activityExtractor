import json
import langid
import sys
reload(sys)
sys.setdefaultencoding('utf8')

def clean(fileName):
    brandList = []
    listFile = open(fileName, 'r')
    for line in listFile:
        if not line.startswith('#'):
            brandList.append(line.strip())
    listFile.close()

    for brand in brandList:
        print(brand)
        userTweetID = {}
        userTweets = {}
        tweetFile = open('data/userTweets2/'+brand+'.json', 'r')
        for line in tweetFile:
            try:
                data = json.loads(line.strip())
            except:
                continue
            userID = data['user']
            if userID not in userTweetID:
                userTweetID[userID] = set()
            if userID not in userTweets:
                userTweets[userID] = []
            tweets = data['statuses']
            for tweet in tweets:
                tweetID = tweet['id']
                if tweetID not in userTweetID[userID]:
                    if len(tweet['text']) > 5:
                        if langid.classify(tweet['text'])[0] == 'en':
                            userTweetID[userID].add(tweetID)
                            userTweets[userID].append(tweet)
        tweetFile.close()

        outputFile = open('data/userTweets2/clean/' + brand + '.json', 'w')
        for userID, tweets in userTweets.items():
            if len(tweets) > 5:
                output = {'user_id': userID, 'statuses': tweets}
                outputFile.write(json.dumps(output)+'\n')
        outputFile.close()


if __name__ == '__main__':
    clean('lists/popularAccount2.list')
