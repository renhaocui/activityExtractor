

def align(modelName, folderList, modelList, weightMode):
    outputData = {}
    for index, model in enumerate(modelList):
        name = 'result/weighted/'+folderList[index]+'/'+model+'_'+modelName+'_'+weightMode+'.sample'
        inputFile = open(name, 'r')
        for line in inputFile:
            data = line.strip().split('\t')
            id = str(data[0])
            if id not in outputData:
                outputData[id] = {'content': data[1], 'trueLabel': data[2], 'place': data[4], model: data[3]}
            else:
                outputData[id][model] = data[3]
        inputFile.close()

    outputFile = open('result/weighted/combined.sample', 'w')
    titleString = 'ID\tContent\tTrueLabel\tPlace\t'
    for model in modelList:
        titleString += model+'\t'
    outputFile.write(titleString.strip()+'\n')

    for id, value in outputData.items():
        if len(value) == 3+len(modelList):
            output = id + '\t' + value['content'] + '\t' + value['trueLabel'] + '\t' + value['place'] + '\t'
            for model in modelList:
                output += value[model] + '\t'
            outputFile.write(output + '\n')
    outputFile.close()


if __name__ == '__main__':
    folderList = ['baselines', 'joint', 'context', 'hierarchical', 'hist=5']
    modelList = ['LSTM', 'J-HistLSTM', 'Context-POSTLSTM', 'H-HistLSTM', 'C-Hist-Context-POST-LSTM']
    align('long1.5', folderList, modelList, 'none')