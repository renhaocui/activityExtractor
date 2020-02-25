import random

def split(modelName, extractIndex):
    dataFile = open('data/consolidateData_' + modelName + '.json', 'r')
    devFile = open('data/consolidateData_'+modelName+'_dev.json', 'w')
    expFile = open('data/consolidateData_'+modelName+'_exp.json', 'w')
    for index, line in enumerate(dataFile):
        if index % 5 == extractIndex:
            devFile.write(line)
        else:
            expFile.write(line)
    dataFile.close()
    devFile.close()
    expFile.close()


def divide(filename, dev=True):
    dataFile = open(filename, 'r')
    data = []
    for line in dataFile:
        data.append(line)
    dataFile.close()

    random.shuffle(data)
    trainFile = open(filename[:-5] + '_train.json', 'w')
    if dev:
        devFile = open(filename[:-5] + '_dev.json', 'w')
    testFile = open(filename[:-5] + '_test.json', 'w')
    for index, line in enumerate(data):
        if dev:
            if index % 5 == 0:
                devFile.write(line)
            elif index % 5 == 1:
                testFile.write(line)
            else:
                trainFile.write(line)
        else:
            if index % 5 == 0:
                testFile.write(line)
            else:
                trainFile.write(line)
    trainFile.close()
    if dev:
        devFile.close()
    testFile.close()



if __name__ == '__main__':
    #split('long1.5', 4)
    divide('data/yelp/consolidateData_yelpUserReview.json', dev=False)
