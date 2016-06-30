import time
import twitter
import json

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

dataFile = open('data', 'w')
#seedWords = ['eating now', 'watching now', 'making now ', 'heading now', 'having now', 'doing now', 'working now']

c_k = '1YjJerf1D5rRvriEzRstKw'
c_s = 'gupHGRRSvQoccMDqbw3slw5KgfV6hYULvjWbr9i2nR4'
a_t = '28816243-o41jYj1BmM40ARLaJzdpJUsosBQhfgah3UJBlrgEU'
a_t_s = 'ZlHbMhluZv0yBlG8cIkMAoFX8XK0pxpFn39OKAqTaQ'

#size = len(seedWords)
index = 0
requestLimit = 180
requestNum = 0
idSet = set()
while True:
    twitter_api = oauth_login()
    requestNum += 1
    if requestNum > requestLimit:
        print 'Wait for 15 mins...'
        time.sleep(900)
        requestNum = 1
    try:
        #print "Collecting Tweets: " + str(seedWords[index])
        print "Collecting Tweets..."
        tweets = twitter_api.search.tweets(language='en', count=100, include_entities=True,
                                           q='"now"')
        #index += 1
        #if index >= size:
        #    index = 0
    except Exception as e:  # take care of errors
        print 'API ERROR: ' + str(e)
        #if index >= size:
        #    index = 0
        continue
    for tweet in tweets['statuses']:
        if tweet['id'] not in idSet:
            idSet.add(tweet['id'])
            temp = {}
            temp['text'] = tweet['text']
            temp['id'] = tweet['id']
            temp['entities'] = tweet['entities']
            temp['geo'] = tweet['geo']
            temp['place'] = tweet['place']
            temp['source'] = tweet['source']
            temp['time'] = tweet['created_at']
            temp['user_lang'] = tweet['user']['lang']
            temp['user_location'] = tweet['user']['location']
            temp['user_timezone'] = tweet['user']['time_zone']
            dataFile.write(json.dumps(temp))
            dataFile.write('\n')

dataFile.close()