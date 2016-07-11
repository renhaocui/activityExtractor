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

dataFile = open('friends', 'a')

c_k = '1YjJerf1D5rRvriEzRstKw'
c_s = 'gupHGRRSvQoccMDqbw3slw5KgfV6hYULvjWbr9i2nR4'
a_t = '28816243-o41jYj1BmM40ARLaJzdpJUsosBQhfgah3UJBlrgEU'
a_t_s = 'ZlHbMhluZv0yBlG8cIkMAoFX8XK0pxpFn39OKAqTaQ'

userInfo = {}
userFile = open('user', 'r')
for line in userFile:
    temp = line.strip().split('\t')
    userInfo[int(temp[0])] = temp[1]
userFile.close()

#size = len(seedWords)
index = 0
requestLimit = 15
requestNum = 0
for userID in userInfo:
    output = {}
    cursor = -1
    twitter_api = oauth_login()
    try:
        print "Collecting friends for: "+str(userID)
        temp = []
        while cursor != 0:
            requestNum += 1
            if requestNum > requestLimit:
                print 'Wait for 15 mins...'
                time.sleep(900)
                requestNum = 1
            response = twitter_api.friends.list(user_id=userID, count=200, cursor=cursor)
            cursor = response['next_cursor']
            for user in response['users']:
                temp.append({'id': user['id'], 'name': user['name'], 'verified': user['verified']})

    except Exception as e:  # take care of errors
        print 'API ERROR: ' + str(e)
        continue
    output[userID] = temp
    dataFile.write(json.dumps(output) + '\n')

dataFile.close()