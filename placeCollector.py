__author__ = 'renhao.cui'
import time
import json
import twitter
import requests

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


def TwitterPlaceCollector():
    requestLimit = 15
    placeList = []
    listFile = open('place.category', 'r')
    for line in listFile:
        placeList.append(line.strip())
    listFile.close()

    twitter_api = oauth_login()
    requestNum = 0
    placeIDSet = set()
    recordFile = open('places.json', 'a')
    for place in placeList:
        print 'extracting data for: ' + place
        try:
            requestNum += 1
            if requestNum > requestLimit:
                print 'Wait for 15 mins...'
                time.sleep(900)
                requestNum = 1
            response = twitter_api.geo.search(query=place, granularity='poi')
            for data in response['result']['places']:
                if data['id'] not in placeIDSet:
                    placeIDSet.add(data['id'])
                    temp = {}
                    temp['name'] = data['full_name']
                    temp['id']=data['id']
                    temp['place'] = place
                    recordFile.write(json.dumps(temp) + '\n')
        except Exception as e:
            print 'API Error: ' + str(e)
            continue
    recordFile.close()


def TomTomPlaceCollector():
    placeList = []
    listFile = open('place.category', 'r')
    for line in listFile:
        placeList.append(line.strip().replace(' ', '+'))
    listFile.close()
    key = 'cx4etag3te8k2d8bss6fy6bd'

    placeIDSet = set()
    recordFile = open('categoryPlaces.json', 'a')
    for place in placeList:
        print 'Collecting [' + place+']'
        for i in range(10):
            offset = 100*i
            serviceURL = 'https://api.tomtom.com/search/2/categorySearch/'+place+'.JSON'+'?limit=100&countrySet=US&ofs='+str(offset)+'&key='+key
            #print serviceURL
            response = requests.get(serviceURL)
            #print response.text
            data = json.loads(response.text)
            for item in data['results']:
                if item['id'] not in placeIDSet:
                    placeIDSet.add(item['id'])
                    temp = {}
                    temp['id'] = item['id']
                    temp['category'] = place.replace('+', ' ')
                    temp['name'] = item['poi']['name']
                    temp['address'] = item['address']['freeformAddress']
                    recordFile.write(json.dumps(temp) + '\n')
            time.sleep(0.2)
            if data['summary']['numResults'] != 100:
                break
    recordFile.close()

if __name__ == '__main__':
    TwitterPlaceCollector()
    #TomTomPlaceCollector()