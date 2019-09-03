import time
__author__ = 'renhao.cui'
import json
import twitter
import properties


def oauth_login():
    # credentials for OAuth
    CONSUMER_KEY = properties.twitter_cred2['c_k']
    CONSUMER_SECRET = properties.twitter_cred2['c_s']
    OAUTH_TOKEN = properties.twitter_cred2['a_t']
    OAUTH_TOKEN_SECRET = properties.twitter_cred2['a_t_s']
    # Creating the authentification
    auth = twitter.oauth.OAuth(OAUTH_TOKEN,
                               OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY,
                               CONSUMER_SECRET)
    # Twitter instance
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def collector(fileName):
    requestLimit = 900
    brandList = []
    listFile = open(fileName, 'r')
    for line in listFile:
        if not line.startswith('#'):
            brandList.append(line.strip())
    listFile.close()

    twitter_api = oauth_login()
    requestNum = 0
    tweetIDSet = set()

    for round in range(100000):
        for brand in reversed(brandList):
            print 'Collecting tweets for: '+str(brand)
            followerFile = open('followers2/'+brand+'.json', 'r')
            outputFile = open('data/userTweets2/'+brand+'.json', 'a')
            for line in followerFile:
                temp = json.loads(line)
                userID = temp['id']
                requestNum += 1
                if requestNum > requestLimit:
                    print 'Wait for 15 mins...'
                    time.sleep(900)
                    requestNum = 1
                try:
                    response = twitter_api.statuses.user_timeline(user_id=userID, count=200, exclude_replies='true', include_rts='false')
                except Exception as e:
                    #print '['+str(userID)+'] '+str(e)
                    continue
                out = {}
                out['user'] = userID
                out['statuses'] = []
                for tweet in response:
                    if tweet['id'] not in tweetIDSet:
                        tempOutput = {}
                        tweetIDSet.add(tweet['id'])
                        tempOutput['id'] = tweet['id']
                        tempOutput['coordinates'] = tweet['coordinates']
                        tempOutput['text'] = tweet['text']
                        tempOutput['place'] = tweet['place']
                        tempOutput['entities'] = tweet['entities']
                        tempOutput['created_at'] = tweet['created_at']
                        tempOutput['geo'] = tweet['geo']
                        out['statuses'].append(tempOutput)
                outputFile.write(json.dumps(out)+'\n')

            outputFile.close()
            followerFile.close()


def extractor(fileName):
    tweetIDSet = set()
    brandList = []
    listFile = open(fileName, 'r')
    for line in listFile:
        brandList.append(line.strip())
    listFile.close()

    for brand in brandList:
        print 'extracting tweets for: '+str(brand)
        tweetFile = open('data/userTweets/'+brand+'.json', 'r')
        outFile = open('data/userTweets/'+brand+'POI.json', 'w')
        for line in tweetFile:
            if len(line) > 50:
                if line.strip().endswith('}'):
                    try:
                        data = json.loads(line.strip())
                    except:
                        continue
                    for tweet in data['statuses']:
                        if tweet['place'] is not None:
                            if tweet['place']['place_type'] == 'poi':
                                if tweet['id'] not in tweetIDSet:
                                    tweet['author'] = data['user']
                                    outFile.write(json.dumps(tweet)+'\n')
                                    tweetIDSet.add(tweet['id'])
        tweetFile.close()
        outFile.close()


def clean(fileName):
    brandList = []
    listFile = open(fileName, 'r')
    for line in listFile:
        if not line.startswith('#'):
            brandList.append(line.strip())
    listFile.close()



if __name__ == '__main__':
    collector('lists/popularAccount2.list')
    #extractor('lists/popularAccount.list')
