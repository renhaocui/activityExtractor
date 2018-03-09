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
        userTweets = {}
        tweetFile = open('data/userTweets2/'+brand+'.json', 'r')
        for line in tweetFile:
            try:
                data = json.loads(line.strip())
            except:
                continue
            userID = data['user']
            if userID not in userTweets:
                userTweets[userID] = []
            tweets = data['statuses']
            for tweet in tweets:
                if len(tweet['text']) > 5:
                    if langid.classify(tweet['text'])[0] == 'en':
                        userTweets[userID].append(tweet)
                if len(userTweets[userID]) > 20:
                    break
        tweetFile.close()

        outputFile = open('data/userTweets2/clean2/' + brand + '.json', 'w')
        for userID, tweets in userTweets.items():
            if len(tweets) > 19:
                output = {'user_id': userID, 'statuses': tweets}
                outputFile.write(json.dumps(output)+'\n')
        outputFile.close()


if __name__ == '__main__':
    clean('lists/popularAccount4.list')
