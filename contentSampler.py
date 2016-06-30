import random
import json

import tweetTextCleaner

inputFile = open('data', 'r')
idFile = open('activityVerifier/tweet.id', 'w')
contentFile = open('activityVerifier/tweet.content', 'w')

contents = {}

for line in inputFile:
    temp = json.loads(line.strip())

    contents[temp['id']] = tweetTextCleaner.tweetCleaner(temp['text'])
inputFile.close()

items = contents.items()
random.shuffle(items)

for id, content in items:
    if not content.startswith('RT : ') and len(content) > 20 and 'What are you' not in content:
        idFile.write(str(id)+'\n')
        contentFile.write(content.encode('utf-8')+'\n')

contentFile.close()
idFile.close()