__author__ = 'renhao.cui'
import time
import json
import twitter

c_k = 'R2FZHZcAcHFatakYhKL2cQcVo'
c_s = 'jwkcIPCkrOBdxKVTVVE7d7cIwH8ZyHHtqxYeCVUZs35Lu4BOkY'
a_t = '141612471-3UJPl93cGf2XBm2JkBn26VFewzwK3WGN1EiKJi4T'
a_t_s = 'do1I1vtIvjgQF3vr0ln4pYVbsAj5OZIxuuATXjgBaqUYM'

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
twitter_api = oauth_login()
idSet = set()
outputFile = open('tweet.json', 'a')
requestNum = 0
while True:
    requestNum += 1
    if requestNum > requestLimit:
        print 'Wait for 15 mins...'
        time.sleep(900)
        requestNum = 1
        print 'Collecting...'
    try:
        response = twitter_api.search.tweets(q='e', count=100, result_type='mixed', lang='en', include_entities='true')
    except Exception as e:
        print 'Error: '+str(e)
        continue
    for tweet in response['statuses']:
        if tweet['place'] is not None:
            if tweet['place']['place_type'] == 'poi':
                if tweet['id'] not in idSet:
                    temp = {}
                    idSet.add(tweet['id'])
                    temp['entities'] = tweet['entities']
                    temp['created_at'] = tweet['created_at']
                    temp['id'] = tweet['id']
                    temp['text'] = tweet['text']
                    temp['place'] = tweet['place']
                    temp['user_location'] = tweet['user']['location']
                    temp['user_id'] = tweet['user']['id']
                    temp['user_description'] = tweet['user']['description']
                    outputFile.write(json.dumps(temp) + '\n')

outputFile.close()