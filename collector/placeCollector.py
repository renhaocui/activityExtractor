__author__ = 'renhao.cui'
import time
import json
import twitter
import requests
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


def TwitterPlaceCollector():
    requestLimit = 15
    placeList = []
    listFile = open('lists/places.list', 'r')
    for line in listFile:
        placeList.append(line.strip())
    listFile.close()

    twitter_api = oauth_login()
    requestNum = 0
    placeIDSet = set()
    recordFile = open('places.json', 'w')
    for place in placeList:
        print('extracting data for: ' + place)
        for i in range(10):
            try:
                requestNum += 1
                if requestNum > requestLimit:
                    print('Wait for 15 mins...')
                    time.sleep(900)
                    requestNum = 1
                response = twitter_api.geo.search(query=place, granularity='poi')
                for data in response['result']['places']:
                    if data['id'] not in placeIDSet:
                        placeIDSet.add(data['id'])
                        temp = {}
                        temp['name'] = data['full_name']
                        temp['id'] = data['id']
                        temp['place'] = place
                        temp['place_type'] = data['place_type']
                        recordFile.write(json.dumps(temp) + '\n')
            except Exception as e:
                print('API Error: ' + str(e))
                continue
    recordFile.close()


def TomTomPlaceCollector():
    placeList = []
    listFile = open('lists/place.tomtom.category', 'r')
    for line in listFile:
        placeList.append(line.strip().replace(' ', '+'))
    listFile.close()
    key = properties.tomtom_cred['key']

    placeIDSet = set()
    recordFile = open('tomtom_category_places.json', 'a')
    for place in placeList:
        print('Collecting [' + place + ']')
        for i in range(20):
            offset = 100 * i
            serviceURL = 'https://api.tomtom.com/search/2/categorySearch/' + place + '.JSON' + '?limit=100&countrySet=US&ofs=' + str(offset) + '&key=' + key
            # print serviceURL
            response = requests.get(serviceURL)
            # print response.text
            data = json.loads(response.text)
            for item in data['results']:
                if item['id'] not in placeIDSet:
                    placeIDSet.add(item['id'])
                    temp = {}
                    temp['id'] = item['id']
                    temp['category'] = place.replace('+', ' ')
                    temp['name'] = item['poi']['name']
                    temp['address'] = item['address']['freeformAddress']
                    temp['lat'] = item['position']['lat']
                    temp['lon'] = item['position']['lon']
                    recordFile.write(json.dumps(temp) + '\n')
            time.sleep(0.2)
            if data['summary']['numResults'] != 100:
                break
    recordFile.close()


def GooglePlaceCollector():
    qList = ['', '0', '1', '2', '3', '4','5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    key = properties.google_cred['key']
    requestLimit = 149000
    categoryList = []
    categoryFile = open('lists/google_place.category', 'r')
    for line in categoryFile:
        if not line.startswith('#'):
            categoryList.append(line.strip())
    categoryFile.close()

    requestNum = 0
    for category in categoryList:
        placeIDSet = set()
        print('Collecting: ' + category)
        nextPageTokens = []
        outputFile = open('data/google_places3/'+category+'.place', 'w')
        for q in qList:
            requestNum += 1
            if requestNum > requestLimit:
                print('wait for 24 hours')
                time.sleep(86400)
                requestNum = 1
            url = 'https://maps.googleapis.com/maps/api/place/textsearch/json?key=' + key + '&language=en&query=' + q + '&type=' + category
            try:
                response = requests.get(url, verify=False)
            except Exception as e:
                print('Error: ' + str(e))
                continue
            data = json.loads(response.text)
            if 'next_page_token' in data:
                if data['next_page_token'] is not None:
                    nextPageTokens.append(data['next_page_token'])
            for item in data['results']:
                if item['id'] not in placeIDSet:
                    placeIDSet.add(item['id'])
                    temp = {}
                    temp['name'] = item['name']
                    temp['lat'] = item['geometry']['location']['lat']
                    temp['lon'] = item['geometry']['location']['lng']
                    temp['id'] = item['id']
                    temp['place_id'] = item['place_id']
                    temp['category'] = item['types']
                    outputFile.write(json.dumps(temp) + '\n')

            furtherTokenList = []
            for token in nextPageTokens:
                requestNum += 1
                if requestNum > requestLimit:
                    print('wait for 24 hours')
                    time.sleep(86400)
                    requestNum = 1
                url = 'https://maps.googleapis.com/maps/api/place/textsearch/json?key=' + key + 'pagetoken=' + token
                try:
                    response = requests.get(url, verify=False)
                except Exception as e:
                    print('Error: ' + str(e))
                    continue
                data = json.loads(response.text)
                if 'next_page_token' in data:
                    if data['next_page_token'] is not None:
                        furtherTokenList.append(data['next_page_token'])
                for item in data['results']:
                    if item['id'] not in placeIDSet:
                        placeIDSet.add(item['id'])
                        temp = {}
                        temp['name'] = item['name']
                        temp['lat'] = item['geometry']['location']['lat']
                        temp['lon'] = item['geometry']['location']['lng']
                        temp['id'] = item['id']
                        temp['place_id'] = item['place_id']
                        temp['category'] = item['types']
                        outputFile.write(json.dumps(temp) + '\n')

            for token in furtherTokenList:
                requestNum += 1
                if requestNum > requestLimit:
                    print('wait for 25 hours')
                    time.sleep(86400)
                    requestNum = 1
                url = 'https://maps.googleapis.com/maps/api/place/textsearch/json?key=' + key + 'pagetoken=' + token
                try:
                    response = requests.get(url, verify=False)
                except Exception as e:
                    print('Error: ' + str(e))
                    continue
                data = json.loads(response.text)
                for item in data['results']:
                    if item['id'] not in placeIDSet:
                        placeIDSet.add(item['id'])
                        temp = {}
                        temp['name'] = item['name']
                        temp['lat'] = item['geometry']['location']['lat']
                        temp['lon'] = item['geometry']['location']['lng']
                        temp['id'] = item['id']
                        temp['place_id'] = item['place_id']
                        temp['category'] = item['types']
                        outputFile.write(json.dumps(temp) + '\n')

        outputFile.close()


if __name__ == '__main__':
    # TwitterPlaceCollector()
    # TomTomPlaceCollector()
    GooglePlaceCollector()
