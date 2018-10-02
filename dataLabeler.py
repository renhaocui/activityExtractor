import json
import re

charLengthLimit = 20
dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}
POSMapper = {'N': 'N', 'O': 'N', '^': 'N', 'S': 'N', 'Z': 'N', 'L': 'N', 'M': 'N',
             'V': 'V', 'A': 'A', 'R': 'R', '@': '@', '#': '#', '~': '~', 'E': 'E', ',': ',', 'U': 'U',
             '!': '0', 'D': '0', 'P': '0', '&': '0', 'T': '0', 'X': '0', 'Y': '0', '$': '0', 'G': '0'}

def hourMapper(hour):
    input = int(hour)
    if 0 <= input < 6:
        output = 0
    elif 6 <= input < 12:
        output = 1
    elif 12 <= input < 18:
        output = 2
    else:
        output = 3
    return output


def list2str(inputList):
    output = ''
    for item in inputList:
        output += item + ' '
    return output.strip().encode('utf-8')


def extractPOS(inputList, breakEmoji=True, removeAllMentions=True):
    posOutput = []
    contentOutput = []
    count = 0.0
    for item in inputList:
        if breakEmoji:
            emojis1 = re.findall(r'\\u....', item[0].encode('unicode-escape'))
            emojis2 = re.findall(r'\\U........', item[0].encode('unicode-escape'))
            emojis = emojis1 + emojis2
            if len(emojis) > 0:
                for emoji in emojis:
                    contentOutput.append(emoji)
                    posOutput.append('E')
            else:
                contentOutput.append(item[0])
                posOutput.append(item[1])
                if '@' in item[1]:
                    count += 1.0
        else:
            contentOutput.append(item[0])
            posOutput.append(item[1])
            if '@' in item[1]:
                count += 1.0
        if len(contentOutput) != len(posOutput):
            print('error')
            print(contentOutput)
            return [], []
    if removeAllMentions:
        if count/len(inputList) > 0.7:
            #print('Remove: ' + list2str(contentOutput))
            return [], []
        else:
            return contentOutput, posOutput
    else:
        return contentOutput, posOutput


def processTweet(modelName, hashtag=False, rules=True):
    activityList = {}
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for index, line in enumerate(activityListFile):
        activityList[index] = line.strip()
    activityListFile.close()

    placeList = {}
    placeListFile = open('lists/google_place_long.category', 'r')
    for index, line in enumerate(placeListFile):
        placeList[index] = line.strip()
    placeListFile.close()

    placeLabelMapper = {}
    labelCount = {}
    placeCount = {}
    for index in range(len(placeList)):
        activity = activityList[index]
        place = placeList[index]
        if (activity != 'NONE') and (not place.startswith('#')):
            placeLabelMapper[place] = activity
            labelCount[activity] = 0.0
            placeCount[place] = 0.0
    #print(placeLabelMapper)

    if rules:
        #labelRule[place][word] = label
        labelRule = {}
        ruleFile = open('lists/mapping.rules', 'r')
        for line in ruleFile:
            temp = line.strip().split('\t')
            tempWords = temp[0].split(';')
            if temp[1] not in labelRule:
                labelRule[temp[1]] = {}
            for tempWord in tempWords:
                pairs = tempWord.split(',')
                labelRule[temp[1]][(pairs[0], pairs[1])] = temp[2]
        ruleFile.close()
        #print(labelRule)

    contents = []
    labels = []
    places = []
    ids = []
    poss = []
    days = []
    hours = []
    created = []
    idTagMapper = {}
    outputFile = open('data/consolidateData_'+modelName+'.json', 'w')
    for place in placeLabelMapper:
        #print('PLACE: '+place)
        if hashtag:
            POSFile = open('data/POS/' + place + '.pos', 'r')
        else:
            POSFile = open('data/POSnew/' + place + '.pos', 'r')
        for line in POSFile:
            data = json.loads(line.strip())
            idTagMapper[data['id']] = data['tag']
        POSFile.close()

        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        for line in tweetFile:
            activity = placeLabelMapper[place]
            data = json.loads(line.strip())
            dateTemp = data['created_at'].split()
            day = dayMapper[dateTemp[0]]
            hour = hourMapper(dateTemp[3].split(':')[0])
            if len(data['text']) > charLengthLimit:
                id = data['id']
                # content = data['text'].replace('\n', ' ').replace('\r', ' ').encode('utf-8').lower()
                contentList, posList = extractPOS(idTagMapper[id], breakEmoji=True)
                if len(contentList) > 2:
                    if rules:
                        if place in labelRule:
                            for index, word in enumerate(contentList):
                                if (word, posList[index]) in labelRule[place]:
                                    #print(list2str(contentList))
                                    activity = labelRule[place][(word, posList[index])]
                    ids.append(id)
                    contents.append(list2str(contentList))
                    poss.append(list2str(posList))
                    labels.append(activity)
                    places.append(place)
                    days.append(day)
                    hours.append(hour)
                    created.append(data['created_at'])
                    labelCount[activity] += 1.0
                    placeCount[place] += 1.0
        tweetFile.close()

    for index, id in enumerate(ids):
        outputFile.write(json.dumps({'id': id, 'label': labels[index], 'place': places[index], 'content': contents[index], 'pos': poss[index], 'day': days[index], 'hour': hours[index], 'created_at': created[index]})+'\n')
    outputFile.close()
    out1 = ''
    out2 = ''
    for label, count in labelCount.items():
        out1 += label + '\t'
        out2 += str(count) + '\t'
    print(out1.strip())
    print(out2.strip())
    print(placeCount)


def processHist(modelName, histNumMin=1, histNumMax=50):
    activityList = {}
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for index, line in enumerate(activityListFile):
        activityList[index] = line.strip()
    activityListFile.close()

    placeList = {}
    placeListFile = open('lists/google_place_long.category', 'r')
    for index, line in enumerate(placeListFile):
        placeList[index] = line.strip()
    placeListFile.close()

    placeLabelMapper = {}
    for index in range(len(placeList)):
        activity = activityList[index]
        place = placeList[index]
        if (activity != 'NONE') and (not place.startswith('#')):
            placeLabelMapper[place] = activity
    #print(placeLabelMapper)

    histTweetData = {}
    for place in placeLabelMapper:
        print(place)
        count = 0
        POShistFile = open('data/POShistCleanMax_50_0.5/' + place + '.pos', 'r')
        idTagMapper = {}
        for line in POShistFile:
            data = json.loads(line.strip())
            idTagMapper[int(data.keys()[0])] = data.values()[0]
        POShistFile.close()
        print(len(idTagMapper))
        histFile = open('data/POIHistClean/' + place + '.json', 'r')
        for line in histFile:
            histData = json.loads(line.strip())
            maxId = histData['max_id']
            if len(histData['statuses']) > histNumMin:
                histTemp = []
                for i in range(min(histNumMax, len(histData['statuses'])-1)):
                    tweet = histData['statuses'][i+1]
                    if tweet['id'] in idTagMapper:
                        if len(idTagMapper[tweet['id']]) > 2:
                            contentList, posList = extractPOS(idTagMapper[int(tweet['id'])], breakEmoji=True)
                            if len(contentList) > 3:
                                dateTemp = tweet['created_at'].split()
                                day = dayMapper[dateTemp[0]]
                                hour = hourMapper(dateTemp[3].split(':')[0])
                                histTemp.append({'id': tweet['id'], 'content': list2str(contentList), 'pos': list2str(posList), 'day': day, 'hour': hour, 'created_at': tweet['created_at']})
                if len(histTemp) > 0:
                    histTweetData[maxId] = histTemp
                    count += 1
        #print(count)
        histFile.close()

    outputFile = open('data/consolidateHistData_' + modelName + '_max.json', 'w')
    for maxId, histTweets in histTweetData.items():
        outputFile.write(json.dumps({maxId: histTweets})+'\n')
    outputFile.close()
    print(len(histTweetData))



def processHist_places(modelName, histNum=5):
    activityList = {}
    activityListFile = open('lists/google_place_activity_' + modelName + '.list', 'r')
    for index, line in enumerate(activityListFile):
        activityList[index] = line.strip()
    activityListFile.close()

    placeList = {}
    placeListFile = open('lists/google_place_long.category', 'r')
    for index, line in enumerate(placeListFile):
        placeList[index] = line.strip()
    placeListFile.close()

    placeLabelMapper = {}
    for index in range(len(placeList)):
        activity = activityList[index]
        place = placeList[index]
        if (activity != 'NONE') and (not place.startswith('#')):
            placeLabelMapper[place] = activity
    #print(placeLabelMapper)

    histTweetData = {}
    for place in placeLabelMapper:
        print(place)
        POShistFile = open('newData/POShistClean/' + place + '.pos', 'r')
        idTagMapper = {}
        for line in POShistFile:
            data = json.loads(line.strip())
            idTagMapper[int(data.keys()[0])] = data.values()[0]
        POShistFile.close()

        histFile = open('newData/POIHistClean/' + place + '.json', 'r')
        for line in histFile:
            histData = json.loads(line.strip())
            maxId = histData['max_id']
            if maxId not in histTweetData:
                histTweetData[maxId] = []
            if len(histData['statuses']) > histNum:
                suffHist = True
                histTemp = []
                for i in range(histNum):
                    tweet = histData['statuses'][i+1]
                    if tweet['id'] not in idTagMapper:
                        suffHist = False
                        break
                    elif len(idTagMapper[tweet['id']]) < 2:
                        suffHist = False
                        break
                    elif tweet['place'] is None:
                        break
                    else:
                        contentList, posList = extractPOS(idTagMapper[tweet['id']], breakEmoji=True)
                        if len(contentList) < 3:
                            suffHist = False
                            break
                    dateTemp = histData['statuses'][i + 1]['created_at'].split()
                    day = dayMapper[dateTemp[0]]
                    hour = hourMapper(dateTemp[3].split(':')[0])
                    histTemp.append({'id': tweet['id'], 'content': list2str(contentList), 'pos': list2str(posList), 'day': day, 'hour': hour, 'created_at': histData['statuses'][i + 1]['created_at'], 'place': tweet['place']})
                if suffHist and len(histTemp) >= histNum:
                    histTweetData[maxId] = histTemp
                elif len(histTweetData[maxId]) == 0:
                    del histTweetData[maxId]
        histFile.close()

    outputFile = open('newData/consolidateHistDataPlaces_' + modelName + '.json', 'w')
    for maxId, histTweets in histTweetData.items():
        outputFile.write(json.dumps({maxId: histTweets})+'\n')
    outputFile.close()
    print(len(histTweetData))


if __name__ == '__main__':
    #processHist('long1.5', histNumMax=50)
    processHist('long1.5', histNumMin=5, histNumMax=5)
    #processTweet('long1.5', hashtag=True)
