import subprocess
import os.path
import tokenizer

#temporalTagger = ternip.recogniser()

def listToStr(inputList):
    output = ''
    for item in inputList:
        output += item + ', '
    return output[:-2]

def loadData(dataFile, labelFile):
    data = []
    for line in dataFile:
        data.append(line.strip())

def POStagger(input):
    if type(input) == list:
        tempFile = open('pos.input', 'w')
        for line in input:
            tempFile.write(line + '\n')
        tempFile.close()
        inputFile = 'pos.input'
    else:
        inputFile = input
    subprocess.check_output(
        'java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp/ark-tweet-nlp-0.3.2.jar --output-format conll ' + inputFile + ' > output.pos',
        shell=True)
    if os.path.exists('pos.input'):
        os.remove('pos.input')
    posFile = open('output.pos', 'r')
    outputTags = []
    sent = []
    for line in posFile:
        temp = line.strip()
        if len(temp) > 0:
            items = temp.split('\t')
            sent.append((items[0], items[1]))
        else:
            outputTags.append(sent)
            sent = []
    return outputTags


def temporalExtractor(input):
    inputList = []
    if len(input[0][0]) > 1:
        for sent in input:
            tempList = []
            for word in sent:
                tempList.append((word[0], word[1], set()))
            inputList.append(tempList)
    else:
        for sent in input:
            tempList = []
            words = tokenizer.tokenize(sent)
            for word in words:
                tempList.append((word, ' ', set()))
            inputList.append(tempList)

    output = temporalTagger.tag(inputList)

    outputList = []
    for sent in output:
        tempList = []
        for word in sent:
            if len(word[2]) > 0:
                tempList.append(word[0])
        outputList.append(tempList)

    return outputList


def NERExtractor():
    inputFile = open('data/experiment/tweet.ner', 'r')
    outputFile = open('data/experiment/tweet.ner.key', 'w')
    output = []
    for line in inputFile:
        entities = ''
        words = line.strip().split(' ')
        firstEnt = True
        for word in words:
            items = word.split('/')
            try:
                if 'B-' in items[1]:
                    if firstEnt:
                        entities += items[1]+': '+items[0]+' '
                        firstEnt = False
                    else:
                        entities += ', ' + items[1]+': ' + items[0] + ' '
                elif 'I-' in items[1]:
                    entities += items[0] + ' '
            except:
                print line
        outputFile.write(entities.strip()+'\n')
        output.append(entities)
    outputFile.close()
    inputFile.close()
    return output

'''
sents = ['This is some annotated samples for the experiment tonight.', 'I have no plan for tomorrow morning.',
         'The game next monday would be awesome!']
POSoutput = POStagger(sents)
print POSoutput
tempOutput = temporalExtractor(sents)
print tempOutput
'''
NERExtractor()