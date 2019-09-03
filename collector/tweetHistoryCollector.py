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


def collector():
    twitter_api = oauth_login()
    requestLimit = 900
    requestNum = 0
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()

    for index, place in enumerate(placeList):
        print(place)
        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        outputFile = open('data/POIhist/' + place + '.json', 'w')
        for line in tweetFile:
            data = json.loads(line.strip())
            id = data['id']
            userID = data['user_id']
            requestNum += 1
            if requestNum > requestLimit:
                print('Wait for 15 mins...')
                time.sleep(900)
                requestNum = 1
            try:
                response = twitter_api.statuses.user_timeline(user_id=userID, count=200, max_id=id, exclude_replies='false', include_rts='true')
            except Exception as e:
                continue
            out = {}
            out['user_id'] = userID
            out['max_id'] = id
            out['statuses'] = []
            for tweet in response:
                tempOutput = {}
                tempOutput['id'] = tweet['id']
                tempOutput['coordinates'] = tweet['coordinates']
                tempOutput['text'] = tweet['text']
                tempOutput['place'] = tweet['place']
                tempOutput['entities'] = tweet['entities']
                tempOutput['created_at'] = tweet['created_at']
                tempOutput['geo'] = tweet['geo']
                tempOutput['lang'] = tweet['lang']
                out['statuses'].append(tempOutput)
            outputFile.write(json.dumps(out) + '\n')

        tweetFile.close()
        outputFile.close()

if __name__ == '__main__':
    collector()
