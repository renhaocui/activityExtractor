from keras.models import model_from_json
import json
from keras.preprocessing import sequence
import numpy as np
import pickle
dayMapper = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}


vocabSize = 10000
tweetLength = 15
#brandList = ['BBCWorld', 'BillGates', 'facebook', 'HillaryClinton', 'instagram', 'justinbieber', 'KDTrey5', 'nytimes', 'Twitter', 'espn', 'google', 'NASA', 'NBA', 'Reuters', 'TheEconomist']

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


def countFreq(inputList):
    count = {}
    for item in inputList:
        if item not in count:
            count[item] = 0
        count[item] += 1
    return count


def npDist2Str(inputList):
    out = ''
    for item in inputList:
        out += str(item*100) + '\t'
    return out.strip()


def followerProcess(brandFileName, structure, modeName, labelNum, balancedWeight='None'):
    brandFile = open(brandFileName, 'r')
    brandList = []
    for line in brandFile:
        brandList.append(line.strip())
    brandFile.close()

    resultFile = open('result/'+structure+'_'+modeName + '_' + str(balancedWeight) + '.resultPercentage', 'w')
    #labelFile = open('model/'+structure+'_'+modeName + '_' + str(balancedWeight) + '.label', 'r')
    #temp = labelFile.read().strip()
    #labels = temp.split()
    print('Loading model...')
    modelFile = open('model/'+structure+'_'+modeName + '_' + str(balancedWeight) + '.json', 'r')
    model_load = modelFile.read()
    modelFile.close()
    model = model_from_json(model_load)
    model.load_weights('model/'+structure+'_'+modeName + '_' + str(balancedWeight) + '.h5')
    tk = pickle.load(open('model/' + structure+'_'+modeName + '_' + str(balancedWeight) + '.tk', 'rb'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('loading data...')
    idSet = set()
    for brand in brandList:
        print(brand)
        userTweet = {}
        tweetCount = 0
        inputFile = open('data/userTweets2/'+brand+'.json', 'r')
        for line in inputFile:
            try:
                data = json.loads(line.strip())
            except:
                continue
            user = data['user']
            if len(data['statuses']) > 0:
                if user not in userTweet:
                    userTweet[user] = []
                for tweet in data['statuses']:
                    if len(tweet['text']) > 10:
                        if tweet['id'] not in idSet:
                            tweetCount += 1
                            idSet.add(tweet['id'])
                            dateTemp = tweet['created_at'].split()
                            userTweet[user].append((tweet['text'].encode('utf-8'), [dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])]))
        inputFile.close()
        #resultFile.write(brand+': '+str(len(userTweet))+'\n')
        #resultFile.write(brand+': '+str(tweetCount)+'\n')

        print('predicting...')
        totalContents = []
        totalTimeList = []
        indexUserMapper = {}

        totalIndex = 0
        for user, tweets in userTweet.items():
            for tweet in tweets:
                totalContents.append(tweet[0])
                totalTimeList.append(tweet[1])
                indexUserMapper[totalIndex] = user
                totalIndex += 1

        timeVector = np.array(totalTimeList)
        tweetSequences = tk.texts_to_sequences(totalContents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
        predictions = model.predict([tweetVector, timeVector])
        #outIndices = np.argmax(predictions, axis=1)
        #tweetSumDistributions = np.sum(predictions, axis=0)

        userTweetDist = {}
        for index, tweetDist in enumerate(predictions):
            user = indexUserMapper[index]
            if user not in userTweetDist:
                userTweetDist[user] = np.zeros([1, labelNum])
            userTweetDist[user] = np.concatenate((userTweetDist[user], [tweetDist]), axis=0)

        userAvgDist = {}
        for user, tweetDist in userTweetDist.items():
            userAvgDist[user] = np.divide(np.sum(tweetDist, axis=0), len(tweetDist)-1)

        accountDist = np.divide(np.sum(userAvgDist.values(), axis=0), len(userAvgDist))

        out = npDist2Str(accountDist)
        resultFile.write(out+'\n')
    resultFile.close()


def tweetProcess(modelNum, labelNum):
    resultFile = open('model/'+modelNum+'.resultTweet', 'w')
    #labelFile = open('model/'+modelNum+'.label', 'r')
    #temp = labelFile.read().strip()
    #labelFile.close()
    print('Loading model...')
    modelFile = open('model/'+modelNum+'.json', 'r')
    model_load = modelFile.read()
    modelFile.close()
    model = model_from_json(model_load)
    model.load_weights('model/'+modelNum+'.h5')

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    print('loading data...')
    for brand in brandList:
        print(brand)
        contents = []
        timeList = []
        idSet = set()
        inputFile = open('data/userTweets/'+brand+'.json', 'r')
        for line in inputFile:
            try:
                data = json.loads(line.strip())
            except:
                continue
            if len(data['statuses']) > 0:
                for tweet in data['statuses']:
                    if tweet['id'] not in idSet:
                        idSet.add(tweet['id'])
                        dateTemp = tweet['created_at'].split()
                        contents.append(tweet['text'].encode('utf-8'))
                        timeList.append([dayMapper[dateTemp[0]], hourMapper(dateTemp[3].split(':')[0])])
        inputFile.close()

        print('predicting...')
        timeVector = np.array(timeList)
        tk = pickle.load(open('model/' + modelNum + '.tk', 'rb'))
        tweetSequences = tk.texts_to_sequences(contents)
        tweetVector = sequence.pad_sequences(tweetSequences, maxlen=tweetLength)
        predictions = model.predict([tweetVector, timeVector])
        outIndices = np.argmax(predictions, axis=1)
        labelFreq = countFreq(outIndices)

        #resultFile.write(brand + ': ' + str(labelFreq) + '\n')
        out = ''
        for i in range(labelNum):
            if i in labelFreq:
                out += str(labelFreq[i]) + '\t'
            else:
                out += '0' + '\t'
        resultFile.write(out.strip()+'\n')
    resultFile.close()



if __name__ == "__main__":
    followerProcess('lists/popularAccount2.list', 'LSTM', 'long1.5', 6, 'none')
    #followerProcess('lists/popularAccount2.list', 'LSTM', 'long1.0', 9, 'class')
    #followerProcess('lists/popularAccount2.list', 'LSTM', 'long1.0', 9, 'sample')