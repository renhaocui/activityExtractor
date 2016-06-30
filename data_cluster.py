import glob
import json
import os
import shutil
import subprocess
import operator
import duplicate_remover
import tweetTextCleaner

def cleanLDAFiles():
    for file in glob.glob('LDA\\LDAinput*.csv.term-counts.cache*'):
        os.remove(file)
    if os.path.exists('LDA\\TMTSnapshots'):
        shutil.rmtree('LDA\\TMTSnapshots')

    return True

def maxIndex(input, num):
    line = {}
    for index in range(len(input)):
        line[index] = float(input[index])
    sorted_line = sorted(line.iteritems(), key=operator.itemgetter(1), reverse=True)
    output = []
    for i in range(num):
        output.append(sorted_line[i][0])
    return output

topicNum = 7
inputFile = open('data', 'r')
csvFile = open('LDA/LDAinput.csv', 'w')
inputData = []
for line in inputFile:
    inputData.append(json.loads(line.strip()))
inputFile.close()

data = duplicate_remover.remove(inputData, 0.3)

textContent = []
for tweet in data:
    text = tweetTextCleaner.tweetCleaner(tweet['text'])
    if not tweetTextCleaner.removeRT(text):
        text = text.encode('utf-8')
        textContent.append(text)
        csvFile.write(text.replace('"', '\'') + '\n')
csvFile.close()

print 'running LDA...'
subprocess.check_output('java -Xmx1024m -jar LDA/tmt-0.4.0.jar LDA/assign.scala', shell=True)

distFile = open('LDA/TMTSnapshots/document-topic-distributions.csv', 'r')

outputFileList = []
for i in range(topicNum):
    outputFile = open('groups/topicGroup'+str(i), 'w')
    outputFileList.append(outputFile)

topicOut = {}
for line in distFile:
    seg = line.strip().split(',')
    if seg[1] != 'NaN':
        topicOutList = maxIndex(seg[1:], 1)
        topicOut[int(seg[0])] = topicOutList
distFile.close()

for index, value in topicOut.items():
    outputFileList[value[0]].write(textContent[index]+'\n')

cleanLDAFiles()

for i in range(topicNum):
    outputFileList[i].close()