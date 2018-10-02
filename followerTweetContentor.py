import json, re, sys
reload(sys)
sys.setdefaultencoding('utf8')


def removeLinks(input):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, 'http://URL')
    return input


def processHistTag():
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()

    idSet = set()
    contents = []
    for index, place in enumerate(placeList):
        print(place)
        histFile = open('data/POIHistClean/' + place + '.json', 'r')
        for line in histFile:
            histData = json.loads(line.strip())
            if len(histData['statuses']) > 1:
                #tweetID = histData['max_id']
                for i in range(len(histData['statuses'])-1):
                    tweet = histData['statuses'][i + 1]
                    content = removeLinks(tweet['text']).replace('\n', ' ').replace('\r', ' ').replace('#', ' #')
                    histID = tweet['id']
                    if histID not in idSet:
                        idSet.add(histID)
                        contents.append(content.strip().encode('utf-8'))

    outputFile = open('data/full.content', 'w')
    for content in contents:
        outputFile.write(content+'\n')
    outputFile.close()

    print len(contents)

if __name__ == '__main__':
    processHistTag()