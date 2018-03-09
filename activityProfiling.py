from keras.models import model_from_json
import json, re, pickle, sys
from keras.preprocessing import sequence
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')

dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}
POSMapper = {'N': 'N', 'O': 'N', '^': 'N', 'S': 'N', 'Z': 'N', 'L': 'N', 'M': 'N',
             'V': 'V', 'A': 'A', 'R': 'R', '@': '@', '#': '#', '~': '~', 'E': 'E', ',': ',', 'U': 'U',
             '!': '0', 'D': '0', 'P': '0', '&': '0', 'T': '0', 'X': '0', 'Y': '0', '$': '0', 'G': '0'}
tweetLength = 25
batch_size = 10

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


def npDist2Str(inputList):
    out = ''
    for item in inputList:
        out += str(item*100) + '\t'
    return out.strip()


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
        if len(inputList) == 0:
            return [], []
        if count/len(inputList) > 0.7:
            #print('Remove: ' + list2str(contentOutput))
            return [], []
        else:
            return contentOutput, posOutput
    else:
        return contentOutput, posOutput


def constructHist(tweets, startIndex, histNum, idMapper):
    outputHistList = []
    for i in range(len(tweets)-startIndex):
        tweet = tweets[startIndex + i]
        if tweet['id'] in idMapper:
            contentList, posList = idMapper[tweet['id']]
            if len(contentList) > 3:
                temp = {}
                temp['content'] = list2str(contentList)
                temp['pos'] = list2str(posList)
                dateTemp = tweet['created_at'].split()
                temp['day'] = dayMapper[dateTemp[0]]
                temp['hour'] = hourMapper(dateTemp[3].split(':')[0])
                outputHistList.append(temp)
        if len(outputHistList) == histNum:
            break
    if len(outputHistList) == histNum:
        return outputHistList
    else:
        return None


def activityPredict(brandFileName, fileName, histNum=3):
    brandFile = open(brandFileName, 'r')
    brandList = []
    for line in brandFile:
        brandList.append(line.strip())
    brandFile.close()

    print('Loading model...')
    modelFile = open(fileName + '_model.json', 'r')
    model_load = modelFile.read()
    modelFile.close()
    model = model_from_json(model_load)
    model.load_weights(fileName + '_model.h5')
    tkTweet = pickle.load(open(fileName + '_tweet.tk', 'rb'))
    tkPOS = pickle.load(open(fileName + '_pos.tk', 'rb'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #print(model.summary())

    resultFile = open('result/accountDist.result', 'w')
    for brand in brandList:
        idMapper = {}
        print('Processing '+brand+'...')

        posFile = open('data/userTweets2/clean2/' + brand + '.pos', 'r')
        for line in posFile:
            data = json.loads(line.strip())
            contentList, posList = extractPOS(data.values()[0], breakEmoji=True)
            if len(contentList) > 3:
                idMapper[int(data.keys()[0])] = (contentList, posList)
        posFile.close()

        tweetData = {}
        tweetFile = open('data/userTweets2/clean2/' + brand + '.json', 'r')
        for line in tweetFile:
            data = json.loads(line.strip())
            tweetData[data['user_id']] = data['statuses']
        tweetFile.close()

        contents = []
        days = []
        hours = []
        poss = []
        histContents = {}
        histDayVectors = {}
        histHourVectors = {}
        histPOSLists = {}
        for i in range(histNum):
            histContents[i] = []
            histDayVectors[i] = []
            histHourVectors[i] = []
            histPOSLists[i] = []
        totalIndex = 0
        indexUserMapper = {}
        for userID, statuses in tweetData.items():
            #print('Tweet #: '+str(len(statuses)))
            for index, tweet in enumerate(statuses):
                if index < (len(statuses)-histNum-3):
                    if tweet['id'] in idMapper:
                        histTweets = constructHist(statuses, index+1, histNum, idMapper)
                        #print(histTweets)
                        if histTweets is None:
                            continue
                        contentList, posList = idMapper[tweet['id']]
                        contents.append(list2str(contentList).encode('utf-8'))
                        poss.append(list2str(posList).encode('utf-8'))
                        dateTemp = tweet['created_at'].split()
                        day = dayMapper[dateTemp[0]]
                        hour = hourMapper(dateTemp[3].split(':')[0])
                        days.append(np.full((tweetLength), day, dtype='int'))
                        hours.append(np.full((tweetLength), hour, dtype='int'))
                        indexUserMapper[totalIndex] = userID
                        totalIndex += 1
                        for i in range(histNum):
                            histContents[i].append(histTweets[i]['content'].encode('utf-8'))
                            histPOSLists[i].append(histTweets[i]['pos'].encode('utf-8'))
                            histDayVectors[i].append(np.full((tweetLength), histTweets[i]['day'], dtype='int'))
                            histHourVectors[i].append(np.full((tweetLength), histTweets[i]['hour'], dtype='int'))

        print('Data size: '+str(len(contents)))
        #print('Valid data#: '+str(len(contents)))
        for i in range(histNum):
            histDayVectors[i] = np.array(histDayVectors[i])
            histHourVectors[i] = np.array(histHourVectors[i])
        days = np.array(days)
        hours = np.array(hours)

        tweetSequences = tkTweet.texts_to_sequences(contents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength, truncating='post', padding='post')
        posSequences = tkPOS.texts_to_sequences(poss)
        posVector = sequence.pad_sequences(posSequences, maxlen=tweetLength, truncating='post', padding='post')

        histTweetVectors = []
        histPOSVectors = []
        for i in range(histNum):
            histDayVectors[i] = np.array(histDayVectors[i])
            histHourVectors[i] = np.array(histHourVectors[i])
            histSequence = tkTweet.texts_to_sequences(histContents[i])
            tempVector = sequence.pad_sequences(histSequence, maxlen=tweetLength, truncating='post', padding='post')
            histTweetVectors.append(tempVector)
            histPOSSequences = tkPOS.texts_to_sequences(histPOSLists[i])
            histPOSVector = sequence.pad_sequences(histPOSSequences, maxlen=tweetLength, truncating='post', padding='post')
            histPOSVectors.append(histPOSVector)

        #print tweetVector.shape
        if len(tweetVector) % batch_size != 0:
            tweetVector = tweetVector[:-(len(tweetVector) % batch_size)]
            days = days[:-(len(days) % batch_size)]
            hours = hours[:-(len(hours) % batch_size)]
            posVector = posVector[:-(len(posVector) % batch_size)]
            for i in range(histNum):
                histTweetVectors[i] = histTweetVectors[i][:-(len(histTweetVectors[i]) % batch_size)]
                histDayVectors[i] = histDayVectors[i][:-(len(histDayVectors[i]) % batch_size)]
                histHourVectors[i] = histHourVectors[i][:-(len(histHourVectors[i]) % batch_size)]
                histPOSVectors[i] = histPOSVectors[i][:-(len(histPOSVectors[i]) % batch_size)]
        #print posVector.shape
        featureList = [tweetVector, days, hours, posVector]
        for i in range(histNum):
            featureList += [histTweetVectors[i], histDayVectors[i], histHourVectors[i], histPOSVectors[i]]
        #print len(featureList)
        predictions = model.predict(featureList, batch_size=batch_size)

        userTweetDist = {}
        for index, tweetDist in enumerate(predictions):
            user = indexUserMapper[index]
            if user not in userTweetDist:
                userTweetDist[user] = np.zeros([1, 6])
            userTweetDist[user] = np.concatenate((userTweetDist[user], [tweetDist]), axis=0)

        userAvgDist = {}
        for user, tweetDist in userTweetDist.items():
            userAvgDist[user] = np.divide(np.sum(tweetDist, axis=0), len(tweetDist) - 1)
        accountDist = np.divide(np.sum(userAvgDist.values(), axis=0), len(userAvgDist))

        out = npDist2Str(accountDist)
        resultFile.write(brand+'\t'+out+'\n')

    resultFile.close()



if __name__ == '__main__':
    activityPredict('lists/popularAccount5.list', 'result/model/C-Hist-Context-POST-LSTM_long1.5_none', histNum=5)