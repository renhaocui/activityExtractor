__author__ = 'renhao.cui'


def longestLength(input):
    outputLength = 0
    for key, value in input.items():
        length = 0
        if value != '-1' and value != '_':
            length += 1
            if value == '0':
                if length > outputLength:
                    outputLength = length
                continue
            nextNode = value
            while nextNode != '-1' and nextNode != '_' and nextNode != '0':
                length += 1
                nextNode = input[nextNode]
        if length > outputLength:
            outputLength = length
    return outputLength


def outputHeads(input):
    output = ''
    for key, value in input.items():
        if value[1] == 0:
            output += value[0] + '/' + value[2] + ' '
    return output.strip()


def keyExtractor():
    parseFile = open('data/experiment/tweet.parse', 'r')
    outputFile = open('data/experiment/tweet.parse.key', 'w')
    forwardData = {}
    backwardData = {}
    index = 0
    for line in parseFile:
        if line.strip() != '':
            words = line.strip().split()
            forwardData[words[0]] = (words[1], words[6], words[4])
            backwardData[words[6]] = (words[1], words[0], words[4])
        else:
            out = ''
            for key, value in forwardData.items():
                if value[2] == 'P':
                    if value[1] in forwardData:
                        if key in backwardData:
                            out += '('+forwardData[value[1]][0] + ' ' + value[0] + ' ' + backwardData[key][0] + '),'
                            if forwardData[value[1]][1] == '0':
                                out = out[:-1] + '*,'
                if value[2] == 'V':
                    if key in backwardData:
                        if backwardData[key][2] == 'N':
                            out += '('+value[0] + ' '+backwardData[key][0]+'),'
                            if value[1] == '0':
                                out = out[:-1] + '*,'
            outputFile.write(out[:-1] + '\n')
            forwardData = {}
            backwardData = {}
            index += 1
    parseFile.close()
    outputFile.close()



def extractor():
    posInputFile = open('dataset/experiment/content/total.posContent.predict', 'r')
    negInputFile = open('dataset/experiment/content/total.negContent.predict', 'r')
    posFile = open('dataset/experiment/ranked/total.pos', 'r')
    negFile = open('dataset/experiment/ranked/total.neg', 'r')
    posLengthFile = open('dataset/experiment/parser/parserLength.pos', 'w')
    negLengthFile = open('dataset/experiment/parser/parserLength.neg', 'w')
    posHeadCountFile = open('dataset/experiment/parser/parserHeadCount.pos', 'w')
    negHeadCountFile = open('dataset/experiment/parser/parserHeadCount.neg', 'w')
    posPOSCountFile = open('dataset/experiment/parser/parserPOSCount.pos', 'w')
    negPOSCountFile = open('dataset/experiment/parser/parserPOSCount.neg', 'w')

    tempData = {}
    tempOutput = {}
    posCount = {'N': 0, 'V': 0, 'A': 0}

    posData = []
    negData = []
    for line in posFile:
        posData.append(line.strip().split(' :: ')[5])
    for line in negFile:
        negData.append(line.strip().split(' :: ')[5])
    posFile.close()
    negFile.close()

    index = 0
    for line in posInputFile:
        if line.strip() != '':
            words = line.strip().split()
            tempData[words[0]] = words[6]
            tempOutput[int(words[0])] = (words[1], int(words[6]), words[4])
            if words[4] in ['N', '^']:
                posCount['N'] += 1
            elif words[4] == 'V':
                posCount['V'] += 1
            elif words[4] in ['A', 'R']:
                posCount['A'] += 1
        else:
            longLen = longestLength(tempData)
            posLengthFile.write(str(longLen) + ' :: ' + posData[index] + '\n')
            posHeadCountFile.write(str(len(outputHeads(tempOutput).split())) + ' :: ' + posData[index] + '\n')
            posPOSCountFile.write(str(posCount['N']) + ' ' + str(posCount['V']) + ' ' + str(posCount['A']) + ' :: ' + posData[index] + '\n')
            tempData = {}
            tempOutput = {}
            posCount = {'N': 0, 'V': 0, 'A': 0}
            index += 1

    tempOutput = {}
    tempData = {}
    posCount = {'N': 0, 'V': 0, 'A': 0}
    index = 0
    for line in negInputFile:
        if line.strip() != '':
            words = line.strip().split()
            tempData[words[0]] = words[6]
            tempOutput[int(words[0])] = (words[1], int(words[6]), words[4])
            if words[4] in ['N', '^']:
                posCount['N'] += 1
            elif words[4] == 'V':
                posCount['V'] += 1
            elif words[4] in ['A', 'R']:
                posCount['A'] += 1
        else:
            longLen = longestLength(tempData)
            negLengthFile.write(str(longLen) + ' :: ' + negData[index] + '\n')
            negHeadCountFile.write(str(len(outputHeads(tempOutput).split())) + ' :: ' + negData[index] + '\n')
            negPOSCountFile.write(str(posCount['N']) + ' ' + str(posCount['V']) + ' ' + str(posCount['A']) + ' :: ' + negData[index] + '\n')

            tempData = {}
            tempOutput = {}
            posCount = {'N': 0, 'V': 0, 'A': 0}
            index += 1

    posInputFile.close()
    negInputFile.close()
    posLengthFile.close()
    negLengthFile.close()
    posHeadCountFile.close()
    negHeadCountFile.close()


if __name__ == '__main__':
    keyExtractor()
