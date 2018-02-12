import subprocess, shlex, json, re
from wordsegment import load, segment
from difflib import SequenceMatcher
import sys
reload(sys)
sys.setdefaultencoding('utf8')
load()

RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx1000m -jar utilities/ark-tweet-nlp-0.3.2.jar"


def removeLinks(input):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, '')
    return input


def _split_results(rows):
    """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
    for line in rows:
        line = line.strip()  # remove '\n'
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                confidence = float(parts[2])
                yield tokens, tags, confidence


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh using a named input file"""

    # remove carriage returns as they are tweet separators for the stdin
    # interface
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)

    # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
    # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
    message = message.encode('utf-8')

    # build a list of args
    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # old call - made a direct call to runTagger.sh (not Windows friendly)
    #po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    # expect a tuple of 2 items like:
    # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
    # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

    pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
    pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
    pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    return pos_result

def combineContents(inputLists):
    contentOutput = []
    for inputList in inputLists:
        content = ''
        for item in inputList:
            content += item[0]
        contentOutput.append(content)
    return contentOutput


def sortTweetbyID(inputList, num=6):
    sorted_list = sorted(inputList, key=lambda k: k['id'], reverse=True)
    return sorted_list[:num]


def processTweetTag(hashtag=True):
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()

    for index, place in enumerate(placeList):
        tweetFile = open('data/POIplace/' + place + '.json', 'r')
        contents = []
        ids = []
        for line in tweetFile:
            data = json.loads(line.strip())
            content = removeLinks(data['text']).replace('\n', ' ').replace('\r', ' ').replace('#', ' #')
            #encode('unicode-escape').replace('\u', ' \u').replace('\U', ' \U')
            if hashtag:
                outContent = content
            else:
                outContent = ''
                for temp in content.split(' '):
                    if temp != '':
                        if temp.startswith('#'):
                            segTemp = segment(temp[1:])
                            for seg in segTemp:
                                outContent += seg + ' '
                        else:
                            outContent += temp + ' '
            contents.append(outContent.strip())
            ids.append(data['id'])
        tweetFile.close()
        print(place + ' size: '+str(len(ids)))

        output = runtagger_parse(contents)
        if len(contents) != len(output):
            print('ERROR')
        else:
            outFile = open('data/POSnew/'+place+'.pos', 'w')
            for index, out in enumerate(output):
                outFile.write(json.dumps({'id': ids[index], 'tag': out})+'\n')
            outFile.close()


def processHistTag(hashtag=True, maxHistNum=10):
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()

    for index, place in enumerate(placeList):
        print(place)
        histFile = open('data/POIHistClean/' + place + '.json', 'r')
        contents = []
        ids = []
        for line in histFile:
            histData = json.loads(line.strip())
            if len(histData['statuses']) > 1:
                #tweetID = histData['max_id']
                for i in range(min(maxHistNum, len(histData['statuses'])-1)):
                    tweet = histData['statuses'][i + 1]
                    content = removeLinks(tweet['text']).replace('\n', ' ').replace('\r', ' ').replace('#', ' #')
                    histID = tweet['id']
                    ids.append(histID)
                    if hashtag:
                        outContent = content
                    else:
                        outContent = ''
                        for temp in content.split(' '):
                            if temp != '':
                                if temp.startswith('#'):
                                    segTemp = segment(temp[1:])
                                    for seg in segTemp:
                                        outContent += seg + ' '
                                else:
                                    outContent += temp + ' '
                    contents.append(outContent.strip())
        outputs = runtagger_parse(contents)

        outFile = open('data/POShistCleanMax_100_0.7/' + place + '.pos', 'w')
        print(str(len(outputs))+'/'+str(len(contents)))
        if len(contents) != len(outputs):
            predictions = combineContents(outputs)
            idIndex = 0
            predIndex = 0
            count = 0
            outputDict = {}
            while True:
                #print idIndex
                #print predIndex
                contentTemp = contents[idIndex].replace(' ', '').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').encode('unicode-escape')
                predTemp = predictions[predIndex].encode('unicode-escape')
                score = SequenceMatcher(None, contentTemp, predTemp).ratio()
                #print contentTemp
                #print predTemp
                #print score
                #print '-------'
                if score > 0.7:
                    outputDict[ids[idIndex]] = outputs[predIndex]
                    count += 1
                    predIndex += 1
                    idIndex += 1
                else:
                    idIndex += 1
                if idIndex >= len(contents) or predIndex >= len(predictions):
                    break
            if count != len(outputs):
                print('ERROR')
            else:
                for key, value in outputDict.items():
                    outFile.write(json.dumps({int(key): value})+'\n')
        else:
            for index, out in enumerate(outputs):
                outFile.write(json.dumps({int(ids[index]): out})+'\n')
        outFile.close()


def processUserTweetTag(fileName, hashtag=True):
    brandList = []
    listFile = open(fileName, 'r')
    for line in listFile:
        if not line.startswith('#'):
            brandList.append(line.strip())
    listFile.close()

    for brand in brandList:
        print(brand)
        userIDList = []
        tweetIDList = []
        contents = []
        inputFile = open('data/userTweets2/clean/' + brand + '.json', 'r')
        outFile = open('data/userTweets2/clean/' + brand + '.pos', 'w')
        for line in inputFile:
            userData = json.loads(line.strip())
            if len(userData['statuses']) > 5:
                user_id = userData['user_id']
                tweets = sortTweetbyID(userData['statuses'], num=6)
                for tweet in tweets:
                    content = removeLinks(tweet['text']).replace('\n', ' ').replace('\r', ' ').replace('#', ' #')
                    tweetID = tweet['id']
                    tweetIDList.append(tweetID)
                    userIDList.append(user_id)
                    if hashtag:
                        outContent = content
                    else:
                        outContent = ''
                        for temp in content.split(' '):
                            if temp != '':
                                if temp.startswith('#'):
                                    segTemp = segment(temp[1:])
                                    for seg in segTemp:
                                        outContent += seg + ' '
                                else:
                                    outContent += temp + ' '
                    contents.append(outContent.strip())
        print ('Running CMU Tagger...')
        outputs = runtagger_parse(contents)

        print ('Aligning tagged outputs...')
        if len(contents) != len(outputs):
            predictions = combineContents(outputs)
            idIndex = 0
            predIndex = 0
            count = 0
            outputDict = {}
            while True:
                contentTemp = contents[idIndex].replace(' ', '').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').encode('unicode-escape')
                predTemp = predictions[predIndex].encode('unicode-escape')
                score = SequenceMatcher(None, contentTemp, predTemp).ratio()
                if score > 0.7:
                    if userIDList[idIndex] not in outputDict:
                        outputDict[userIDList[idIndex]] = [{'id': tweetIDList[idIndex], 'tag': outputs[predIndex]}]
                    else:
                        outputDict[userIDList[idIndex]].append({'id': tweetIDList[idIndex], 'tag': outputs[predIndex]})
                    count += 1
                    predIndex += 1
                    idIndex += 1
                else:
                    idIndex += 1
                if idIndex >= len(contents) or predIndex >= len(predictions):
                    break
            if count != len(outputs):
                print('ERROR')
            else:
                for userID, value in outputDict.items():
                    outFile.write(json.dumps({userID: value})+'\n')
        else:
            outDict = {}
            for index, out in enumerate(outputs):
                userID = userIDList[index]
                if userID not in outDict:
                    outDict[userID] = [{'id': tweetIDList[index], 'tag': out}]
                else:
                    outDict[userID].append({'id': tweetIDList[index], 'tag': out})
            for userID, value in outDict:
                outFile.write(json.dumps({userID: value})+'\n')
        outFile.close()



if __name__ == "__main__":
    #processTweetTag(hashtag=False)
    processHistTag(hashtag=False, maxHistNum=100)
    #processUserTweetTag('lists/popularAccount2.list', hashtag=False)
