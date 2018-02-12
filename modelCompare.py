import os
from difflib import SequenceMatcher
import sys
reload(sys)
sys.setdefaultencoding('utf8')


def matchContent(contentList, inputContent, ratio):
    if len(contentList) == 0:
        return inputContent
    for content in contentList:
        score = SequenceMatcher(None, content, inputContent).ratio()
        if score > ratio:
            return content
    return inputContent


def allMatch(inputDict):
    flagLabel = inputDict['LSTM']
    for key, value in inputDict.items():
        if key not in ['content', 'Location', 'TrueLabel']:
            if value != flagLabel:
                return False
    return True


def oneCorrect(inputDict):
    trueLabel = inputDict['TrueLabel']
    for key, value in inputDict.items():
        if key not in ['content', 'Location', 'TrueLabel']:
            if value == trueLabel:
                return True
    return False


def compare(inputFolder, weightMode='none'):
    modelData = {}
    modelNames = []
    for file in os.listdir(inputFolder):
        if file.endswith(weightMode+'.sample'):
            modelName = file.split('_')[0]
            modelNames.append(modelName)
            sampleFile = open(inputFolder+'/'+file, 'r')
            for line in sampleFile:
                data = line.strip().split('\t')
                #content = matchContent(modelData.keys(), data[0], 0.9)
                id = data[0]
                if id not in modelData:
                    modelData[id] = {}
                modelData[id]['content'] = data[1]
                modelData[id]['TrueLabel'] = data[2]
                modelData[id][modelName] = data[3]
                modelData[id]['Location'] = data[4]
            sampleFile.close()

    modelList = list(set(modelNames))
    outputFile = open(inputFolder+'/comparison.sample', 'w')
    tempStr = 'ID\tContent\tLocation\tTrueLabel'
    for name in modelList:
        tempStr += '\t' + name
    outputFile.write(tempStr.strip() + '\n')
    for id, items in modelData.items():
        if len(items) > 15:
            if not allMatch(items):
                if oneCorrect(items):
                    outStr = id + '\t' + items['content'] + '\t'+items['Location']+'\t'+items['TrueLabel']
                    for name in modelList:
                        if name in items:
                            outStr += '\t' + items[name]
                        else:
                            outStr += '\t' + 'None'
                    outputFile.write(outStr.strip() + '\n')
    outputFile.close()


if __name__ == '__main__':
    compare('samples_full', weightMode='none')