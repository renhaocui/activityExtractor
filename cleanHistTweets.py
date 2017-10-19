import json
import langid

placeList = []
placeListFile = open('lists/google_place_long.category', 'r')
for line in placeListFile:
    if not line.startswith('#'):
        placeList.append(line.strip())
placeListFile.close()

for index, place in enumerate(placeList):
    print place
    outputFile = open('data/POIHistClean/' + place + '.json', 'w')
    histFile = open('data/POIHist/' + place + '.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        max_id = data['max_id']
        user_id = data['user_id']
        statuses = data['statuses']
        outputStatuses = []
        for tweet in statuses:
            if len(tweet['text']) > 1:
                if langid.classify(tweet['text'])[0] == 'en':
                    outputStatuses.append(tweet)
        output = {'max_id': max_id, 'user_id': user_id, 'statuses': outputStatuses}
        outputFile.write(json.dumps(output)+'\n')
    histFile.close()
    outputFile.close()

