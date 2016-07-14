import time

__author__ = 'renhao.cui'
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


requestLimit = 15
brandList = []
listFile = open('brand.list', 'r')
for line in listFile:
    brandList.append(line.strip())
listFile.close()

twitter_api = oauth_login()
requestNum = 0
for brand in brandList:
    followerIDSet = set()
    print 'extracting followers for: ' + brand
    recordFile = open("followers//" + brand + '.json', 'a')
    cursor = -1
    for i in range(10000):
        if cursor == 0 or len(followerIDSet) > 50000:
            break
        try:
            requestNum += 1
            if requestNum > requestLimit:
                print 'Wait for 15 mins...'
                time.sleep(900)
                requestNum = 1
            response = twitter_api.followers.list(screen_name=brand, include_user_entities='true', count=200, cursor=cursor, skip_status='false')
            cursor = response['next_cursor']
            for data in response['users']:
                if data['id'] not in followerIDSet:
                    followerIDSet.add(data['id'])
                    temp = {}
                    temp['id'] = data['id']
                    temp['verified'] = data['verified']
                    temp['entities'] = data['entities']
                    temp['description'] = data['description']
                    temp['follower_count'] = data['follower_count']
                    temp['listed_count'] = data['listed_count']
                    temp['status'] = data['status']
                    temp['status_count'] = data['status_count']
                    temp['location'] = data['location']
                    temp['friends_count'] = data['friends_count']
                    temp['screen_name'] = data['screen_name']
                    temp['lang'] = data['lang']
                    temp['favourites_count'] = data['favourites_count']
                    recordFile.write(json.dumps(temp)+'\n')
        except Exception as e:
            print 'API Error: '+str(e)
            continue
    recordFile.close()
