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


def divide(modelName):
    dataFile = open('data/consolidateData_' + modelName + '.json', 'r')
    data = []
    for line in dataFile:
        data.append(line)
    dataFile.close()

    random.shuffle(data)
    trainFile = open('data/consolidateData_' + modelName + '_train.json', 'w')
    devFile = open('data/consolidateData_' + modelName + '_dev.json', 'w')
    testFile = open('data/consolidateData_' + modelName + '_test.json', 'w')
    for index, line in enumerate(data):
        if index % 5 == 0:
            devFile.write(line)
        elif index % 5 == 1:
            testFile.write(line)
        else:
            trainFile.write(line)
    trainFile.close()
    devFile.close()
    testFile.close()



if __name__ == '__main__':
    #split('long1.5', 4)
    divide('long1.5')
