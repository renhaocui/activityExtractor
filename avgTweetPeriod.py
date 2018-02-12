import json, time, datetime, sys
import statistics
from wordsegment import load
reload(sys)
sys.setdefaultencoding('utf8')
load()

def process(modelName, periodNum=3):
    print('Loading...')
    histDataSet = {}
    histFile = open('data/consolidateHistData_' + modelName + '_max.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        histDataSet[int(data.keys()[0])] = data.values()[0]
    histFile.close()

    countData = []
    dataFile = open('data/consolidateData_' + modelName + '_CreatedAt.json', 'r')
    for line in dataFile:
        data = json.loads(line.strip())
        if data['id'] in histDataSet:
            histTweets = histDataSet[data['id']]
            timeTemp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(data['created_at'],'%a %b %d %H:%M:%S +0000 %Y'))
            createdTimestamp = time.mktime(datetime.datetime.strptime(timeTemp, '%Y-%m-%d %H:%M:%S').timetuple())
            count = 0
            for histTweet in reversed(histTweets):
                timeTemp = time.strftime('%Y-%m-%d %H:%M:%S', time.strptime(histTweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
                histCreatedTimestamp = time.mktime(datetime.datetime.strptime(timeTemp, '%Y-%m-%d %H:%M:%S').timetuple())
                if (createdTimestamp - histCreatedTimestamp) < periodNum * 8 * 3600:
                    count += 1
            countData.append(count)
    print(len(countData))
    print(countData.count(0))
    print(statistics.mean(countData))
    print(statistics.stdev(countData))


if __name__ == '__main__':
    for num in [3, 6, 9]:
        print('Period: '+str(num))
        process('long1.5', periodNum=num)
