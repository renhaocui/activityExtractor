__author__ = 'renhao.cui'
import time
import json
import twitter
import os
'''
c_k = 'wHqszw01omK3W7sNwZC2XgY2e'
c_s = '2Kmt5CFVG8UikLLKNgTUbertPfxOBHSHaqZDdMZ5T6vgP11iD8'
a_t = '141612471-rtZFDyJrcaLN96FYpTSRyoCyhMcFySLZCTA2VXXF'
a_t_s = 'zYUlpJTApBhtgnAP1PpypO8TCofZdqIGb9CZO6o5Z8vUA'
'''
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

def tomtomTweetCollector():
    requestLimit = 180
    placeData = {}
    listFile = open('categoryPlaces2.json', 'r')
    for line in listFile:
        data = json.loads(line.strip())
        category = data['category']
        id = data['id']
        address = data['address']
        name = data['name']
        lat = data['lat']
        lon = data['lon']
        if category not in placeData:
            placeData[category] = {id: {'address': address, 'name': name, 'lat': lat, 'lon': lon}}
        else:
            placeData[category][id] = {'address': address, 'name': name, 'lat': lat, 'lon': lon}
    listFile.close()

    twitter_api = oauth_login()
    requestNum = 0
    outData = {}
    #response = twitter_api.search.tweets(q=' ', geocode='35.23569,-88.38866,0.1mi', result_type='mixed', count=100, lang='en')
    #print response

    for category in placeData:
        tweetIDSet = set()
        print 'Collecting for ['+category+']'
        outFile = open('data/' + category + '.json', 'w')
        for id in placeData[category]:
            requestNum += 1
            if requestNum > requestLimit:
                print 'Wait for 15 mins...'
                time.sleep(900)
                requestNum = 1
            try:
                response = twitter_api.search.tweets(q=' ', geocode=str(placeData[category][id]['lat'])+','+str(placeData[category][id]['lon'])+','+'0.1mi', count=100, result_type='mixed', lang='en')
            except Exception as e:
                print 'Error ' + str(e)
                continue
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
                    temp['place_name'] = placeData[category][id]['name']
                    temp['place_category'] = category
                    temp['user_location'] = tweet['user']['location']
                    temp['user_id'] = tweet['user']['id']
                    temp['user_description'] = tweet['user']['description']
                    outFile.write(json.dumps(temp)+'\n')

        outFile.close()


def googleTweetCollector(occurrence):
    requestLimit = 180
    requestNum = 0
    twitter_api = oauth_login()
    tweetIDSet = set()
    for i in range(occurrence):
        print 'Round '+str(i)
        for file in os.listdir("data/google_places3"):
            print 'Collecting: '+file.split('.')[0]
            placeData = {}
            inputFile = open('data/google_places3/'+file, 'r')
            for line in inputFile:
                data = json.loads(line.strip())
                lon = data['lon']
                lat = data['lat']
                category = data['category']
                name = data['name']
                id = data['id']
                placeData[id] = {'name': name, 'lon': lon, 'lat': lat, 'category': category}
            inputFile.close()

            outputFile = open('data/google_place_tweets3.2/'+file.split('.')[0] + '.json', 'a')
            for id in placeData:
                requestNum += 1
                if requestNum > requestLimit:
                    print 'Wait for 15 mins...'
                    time.sleep(900)
                    requestNum = 1
                try:
                    response = twitter_api.search.tweets(q=' ', geocode=str(placeData[id]['lat']) + ',' + str(placeData[id]['lon']) + ',' + '0.01km', count=100, result_type='mixed',
                                                         lang='en')
                except Exception as e:
                    print 'Error ' + str(e)
                    continue
                for tweet in response['statuses']:
                    if tweet['id'] not in tweetIDSet:
                        temp = {}
                        tweetIDSet.add(tweet['id'])
                        temp['entities'] = tweet['entities']
                        temp['created_at'] = tweet['created_at']
                        temp['id'] = tweet['id']
                        temp['text'] = tweet['text']
                        temp['place'] = tweet['place']
                        temp['google_place_name'] = placeData[id]['name']
                        temp['google_place_category'] = placeData[id]['category']
                        temp['user_location'] = tweet['user']['location']
                        temp['user_id'] = tweet['user']['id']
                        temp['user_description'] = tweet['user']['description']
                        outputFile.write(json.dumps(temp) + '\n')
            outputFile.close()
        print 'Wait for 8 hours...'
        time.sleep(28800)

if __name__ == '__main__':
    #tomtomTweetCollector()
    googleTweetCollector(10)