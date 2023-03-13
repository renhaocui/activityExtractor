import time
import json
import requests

key = 'AIzaSyCx4TvgtOlzVV33dzeejTI5G8g23xDUCYw'
requestLimit = 5000
categoryList = ['cafe', 'bakery']

def nearbySearch(latitude,longitude):
    requestNum = 0
    for category in categoryList:
        placeIDSet = set()
        nextPageTokens = []
        outputFile = open('data/'+category+'.place', 'w')
        requestNum += 1
        if requestNum > requestLimit:
            print('wait for 24 hours')
            time.sleep(86400)
            requestNum = 1
        url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={key}&language=en&&type={category}&radius=400&location={latitude},{longitude}'
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
            url = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={key}&pagetoken={token}'
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


def parse(inputString):
    data = json.loads(inputString)
    for item in data['results'][][][][]:
        print(item['value'])


if __name__ == '__main__':
    #nearbySearch()
    inputString = '{"status": "ok", "results": [{"name": "123", "value": "0.123"}, {"name": "234", "value": "0.234"}]}'
    '{"status": "ok"}'
    parse(inputString)