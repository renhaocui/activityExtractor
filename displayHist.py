import json
import sys
reload(sys)
sys.setdefaultencoding('utf8')


def display(modelName, histNum, idListFile):
    idList = []
    idFile = open(idListFile, 'r')
    for line in idFile:
        idList.append(str(line.strip()))
    idFile.close()
    print idList

    histData = {}
    histFile = open('data/consolidateHistData_' + modelName + '.json', 'r')
    for line in histFile:
        data = json.loads(line.strip())
        if str(data.keys()[0]) in idList:
            histData[str(data.keys()[0])] = data.values()[0]
    histFile.close()
    #print histData

    histContents = {}
    for i in range(histNum):
        histContents[i] = []
    dataFile = open('data/consolidateData_' + modelName + '.json', 'r')
    outputFile = open('result/histDisplay.list', 'w')
    for line in dataFile:
        data = json.loads(line.strip())
        if str(data['id']) in histData:
            histTweets = histData[str(data['id'])]
            if len(histTweets) >= histNum:
                output = str(data['id']) + '\t' + data['content'].encode('utf-8')
                for i in range(histNum):
                    output += '\t' + histTweets[i]['content'].encode('utf-8')
                outputFile.write(output+'\n')
    outputFile.close()

if __name__ == '__main__':
    display('long1.5', 5, 'result/sampleHist.list')
