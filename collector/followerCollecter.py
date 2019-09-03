__author__ = 'renhao.cui'
import time
import json
import twitter
import properties


def oauth_login():
    # credentials for OAuth
    CONSUMER_KEY = properties.twitter_cred['c_k']
    CONSUMER_SECRET = properties.twitter_cred['c_s']
    OAUTH_TOKEN = properties.twitter_cred['a_t']
    OAUTH_TOKEN_SECRET = properties.twitter_cred['a_t_s']
    # Creating the authentification
    auth = twitter.oauth.OAuth(OAUTH_TOKEN,
                               OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY,
                               CONSUMER_SECRET)
    # Twitter instance
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api


def collector(fileName):
    requestLimit = 15
    brandList = []
    listFile = open(fileName, 'r')
    for line in listFile:
        if not line.startswith('#'):
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
            if cursor == 0 or len(followerIDSet) > 10000:
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
                    if data['lang'] == 'en':
                        if data['id'] not in followerIDSet:
                            followerIDSet.add(data['id'])
                            temp = {}
                            temp['id'] = data['id']
                            temp['verified'] = data['verified']
                            temp['entities'] = data['entities']
                            temp['description'] = data['description']
                            temp['followers_count'] = data['followers_count']
                            temp['listed_count'] = data['listed_count']
                            #temp['status'] = data['status']
                            temp['statuses_count'] = data['statuses_count']
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

if __name__ == '__main__':
    collector('lists/popularAccount.list')
