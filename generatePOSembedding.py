import subprocess, shlex, json, re, gensim
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

RUN_TAGGER_CMD = "java -XX:ParallelGCThreads=2 -Xmx1000m -jar utilities/ark-tweet-nlp-0.3.2.jar"

def removeLinks(input):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, '')
    return input


def _split_results(rows):
    for line in rows:
        line = line.strip()
        if len(line) > 0:
            if line.count('\t') == 2:
                parts = line.split('\t')
                tokens = parts[0]
                tags = parts[1]
                confidence = float(parts[2])
                yield tokens, tags, confidence


def _call_runtagger(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
    message = "\n".join(tweets_cleaned)
    message = message.encode('utf-8')

    args = shlex.split(run_tagger_cmd)
    args.append('--output-format')
    args.append('conll')
    po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = po.communicate(message)
    pos_result = result[0].strip('\n\n')
    pos_result = pos_result.split('\n\n')
    pos_results = [pr.split('\n') for pr in pos_result]
    return pos_results


def runtagger_parse(tweets, run_tagger_cmd=RUN_TAGGER_CMD):
    pos_raw_results = _call_runtagger(tweets, run_tagger_cmd)
    pos_result = []
    for pos_raw_result in pos_raw_results:
        pos_result.append([x for x in _split_results(pos_raw_result)])
    return pos_result


def loadData(brandFileName, scoreLimit):
    brandFile = open(brandFileName, 'r')
    brandList = []
    for line in brandFile:
        brandList.append(line.strip())
    brandFile.close()

    print('Loading data...')
    idSet = set()
    tweetCount = 0
    tweets = []
    for brand in brandList:
        print(brand)
        inputFile = open('data/userTweets2/' + brand + '.json', 'r')
        for line in inputFile:
            try:
                data = json.loads(line.strip())
            except:
                continue
            if len(data['statuses']) > 0:
                for tweet in data['statuses']:
                    if len(tweet['text']) > 10:
                        if tweet['id'] not in idSet:
                            tweetCount += 1
                            idSet.add(tweet['id'])
                            tweets.append(removeLinks(tweet['text']).encode('utf-8'))
        inputFile.close()
    print('total count: '+str(tweetCount))

    print('Running tagger...')
    count = 0
    output = runtagger_parse(tweets)
    posList = []
    for index, out in enumerate(output):
        valid = True
        posOut = []
        for item in out:
            posOut.append(item[1])
            if float(item[2]) < scoreLimit:
                valid = False
                break
        if valid:
            count += 1
            posList.append(posOut)
        posList.append(posOut)
    print('valid count: '+str(count))
    return posList


def train(input):
    print('training embedding...')
    model = gensim.models.Word2Vec(input, size=200, min_count=10, workers=4)
    model.save('../tweetEmbeddingData/pos.word2vec')
    print('Done')

if __name__ == '__main__':
    data = loadData('lists/popularAccount2.list', 0.7)
    #data = loadData('lists/3.list', 0.7)
    train(data)