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


requestLimit = 2
placeList = []
listFile = open('place.list', 'r')
for line in listFile:
    placeList.append(line.strip())
listFile.close()

twitter_api = oauth_login()
requestNum = 0
placeIDSet = set()
recordFile = open('places.json', 'a')
for place in placeList:
    print 'extracting followers for: ' + place
    try:
        requestNum += 1
        if requestNum > requestLimit:
            print 'Wait for 15 mins...'
            time.sleep(900)
            requestNum = 1
        response = twitter_api.geo.search(query=place, granularity='poi')
        print response
        for data in response['result']['places']:
            if data['id'] not in placeIDSet:
                placeIDSet.add(data['id'])
                temp = {}
                temp['namne'] = data['full_name']
                temp['id']=data['id']
                temp['place'] = place
                recordFile.write(json.dumps(temp) + '\n')
    except Exception as e:
        print 'API Error: ' + str(e)
        continue
recordFile.close()
