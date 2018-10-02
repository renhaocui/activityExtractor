import json

idSet = set()

def clean(modelName):
    inputFile = open('data/consolidateData_'+modelName+'.json', 'r')
    outputFile = open('data/consolidateData_'+modelName+'_clean.json', 'w')
    for line in inputFile:
        data = json.loads(line.strip())
        id = data['id']
        if id not in idSet:
            idSet.add(data['id'])
            outputFile.write(line)
    inputFile.close()
    outputFile.close()


if __name__ == '__main__':
    clean('long1.5')