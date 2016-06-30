__author__ = 'rencui'
import json
import tokenizer

def lcsLen(a, b):
    table = [[0] * (len(b) + 1) for _ in xrange(len(a) + 1)]
    for i, ca in enumerate(a, 1):
        for j, cb in enumerate(b, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def validTweet(tweetData):
    if len(tweetData['entities']['hashtags']) > 5:
        return False
    elif len(tweetData['entities']['urls']) > 5:
        return False
    else:
        return True


def remove(inputJsonList, filterRatio):
    idSet = set()
    index = 0
    tempSeg1 = []
    tempSeg2 = []
    tempSeg3 = []
    output = []

    for data in inputJsonList:
        id = data['id']
        if id not in idSet:
            content = data['text'].replace('\n', ' ')
            seg = tokenizer.simpleTokenize(content.lower())
            if len(seg) > 10:
                if abs(len(seg) - len(tempSeg1)) < 3:
                    llcs = lcsLen(seg, tempSeg1)
                    total = float(len(seg))
                    ratio = llcs / total
                    if ratio > filterRatio:
                        index += 1
                        tempSeg3 = tempSeg2
                        tempSeg2 = tempSeg1
                        tempSeg1 = seg
                        continue
                if abs(len(seg) - len(tempSeg2)) < 3:
                    llcs = lcsLen(seg, tempSeg2)
                    total = float(len(seg))
                    ratio = llcs / total
                    if ratio > filterRatio:
                        index += 1
                        tempSeg3 = tempSeg2
                        tempSeg2 = tempSeg1
                        tempSeg1 = seg
                        continue
                if abs(len(seg) - len(tempSeg3)) < 3:
                    llcs = lcsLen(seg, tempSeg3)
                    total = float(len(seg))
                    ratio = llcs / total
                    if ratio > filterRatio:
                        index += 1
                        tempSeg3 = tempSeg2
                        tempSeg2 = tempSeg1
                        tempSeg1 = seg
                        continue
                tempSeg3 = tempSeg2
                tempSeg2 = tempSeg1
                tempSeg1 = seg
                try:
                    if validTweet(data):
                        output.append(data)
                        idSet.add(id)
                except Exception as ex:
                    print '[Error:  ' + str(ex) + '] ' + str(id) + '\n'
                    continue
    return output