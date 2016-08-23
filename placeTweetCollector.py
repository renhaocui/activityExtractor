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
placeData = {}
listFile = open('categoryPlaces.json', 'r')
for line in listFile:
    data = json.loads(line.strip())
    category = data['category']
    id = data['id']
    address = data['address']
    name = data['name']
    if category not in placeData:
        placeData[category] = {id: {'address': address, 'name': name}}
    else:
        placeData[category][id] = {'address': address, 'name': name}
listFile.close()

twitter_api = oauth_login()
requestNum = 0
outData = {}
response = twitter_api.search.tweets(q='place:12f0133ced57367b')
print response

'''
for category in placeData:
    tweetIDSet = set()
    print 'Collecting for ['+category+']'
    outFile = open('data/' + category + '.json', 'w')
    for id in placeData[category]:
        print id
        requestNum += 1
        outFile = open('data/'+category+'.json', 'a')
        if requestNum > requestLimit:
            print 'Wait for 15 mins...'
            time.sleep(900)
            requestNum = 1
        response = twitter_api.search.tweets(q='place:tomtom:'+str(id), count=100, result_type='mixed')
        for tweet in response['statuses']:
            if tweet['id'] not in tweetIDSet:
                temp = {}
                tweetIDSet.add(tweet['id'])
                temp['entities'] = tweet['entities']
                temp['created_at'] = tweet['created_at']
                temp['id'] = tweet['id']
                temp['text'] = tweet['text']
                temp['retweet_count'] = tweet['retweet_count']
                temp['place'] = tweet['place']
                temp['place_category'] = category
                temp['user_location'] = tweet['user']['location']
                temp['user_id'] = tweet['user']['id']
                temp['user_description'] = tweet['user']['description']
                outFile.write(json.dumps(temp)+'\n')

    outFile.close()
'''