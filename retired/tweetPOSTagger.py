import subprocess
import os.path

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
    wordTag = {}
    for line in posFile:
        temp = line.strip()
        if len(temp) > 0:
            items = temp.split('\t')
            wordTag[items[0]] = items[1]
        else:
            outputTags.append(wordTag)
            wordTag = {}
    return outputTags

print POStagger('data/casual.txt')
print POStagger(['This is a test!', 'test tweet #justfortest', 'hello world! yep'])
