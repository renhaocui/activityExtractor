import time
__author__ = 'renhao.cui'
import json
import twitter

c_k = 'wHqszw01omK3W7sNwZC2XgY2e'
c_s = '2Kmt5CFVG8UikLLKNgTUbertPfxOBHSHaqZDdMZ5T6vgP11iD8'
a_t = '141612471-rtZFDyJrcaLN96FYpTSRyoCyhMcFySLZCTA2VXXF'
a_t_s = 'zYUlpJTApBhtgnAP1PpypO8TCofZdqIGb9CZO6o5Z8vUA'


def oauth_login():
    # credentials for OAuth
    CONSUMER_KEY = c_k
    CONSUMER_SECRET = c_s
    OAUTH_TOKEN = a_t
    OAUTH_TOKEN_SECRET = a_t_s
    # Creating the authentification
    auth = twitter.oauth.OAuth(OAUTH_TOKEN,
                               OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY,
                               CONSUMER_SECRET)
    # Twitter instance
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


requestLimit = 180
tweetLimit = 100000
brandList = []
listFile = open('brand.list', 'r')
for line in listFile:
    brandList.append(line.strip())
listFile.close()

twitter_api = oauth_login()
requestNum = 0

for brand in brandList:
    print 'Collecting tweets for: '+str(brand)
    tweetIDSet = set()
    followerFile = open('followers/'+brand+'.json', 'r')
    outputFile = open('tweets/'+brand+'.json', 'a')
    for line in followerFile:
        if len(tweetIDSet) > tweetLimit:
            break
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
            print '['+str(userID)+'] '+str(e)
            continue
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
                tempOutput['user'] = userID
                outputFile.write(json.dumps(tempOutput)+'\n')

    outputFile.close()
    followerFile.close()
