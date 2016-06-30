import subprocess


def POStagger():
    subprocess.check_output('java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp/ark-tweet-nlp-0.3.2.jar --output-format conll ark-tweet-nlp/examples/casual.txt > output.pos', shell=True)
    posFile = open('output.pos', 'r')
    sentenceTags = {}
    index = 0
    wordTag = {}
    for line in posFile:
        temp = line.strip()
        if len(temp) > 0:
            items = temp.split('\t')
            wordTag[items[0]] = items[1]
        else:
            sentenceTags[index] = wordTag
            index += 1
            wordTag = {}

    return sentenceTags

POStagger()