import json
import re

from utilities import tweetTextCleaner

regex_str = [
    r'<[^>]+>',  # HTML tags
    r'(?:@[\w_]+)',  # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)",  # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # URLs

    r'(?:(?:\d+,?)+(?:\.?\d+)?)',  # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])",  # words with - and '
    r'(?:[\w_]+)',  # other words
    r'(?:\S)+'  # anything else
]

tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE)


def tokenize(s):
    return tokens_re.findall(s)


def run():
    placeList = []
    placeListFile = open('lists/google_place_long.category', 'r')
    for line in placeListFile:
        if not line.startswith('#'):
            placeList.append(line.strip())
    placeListFile.close()

    for place in placeList:
        tweetFile = open('data/google_place_tweets3.3/POI/' + place + '.json', 'r')
        cleanTweetFile = open('data/google_place_tweets3.3/POI_clean/' + place + '.json', 'w')
        for line in tweetFile:
            data = json.loads(line.strip())
            content = data['text'].replace('\n', ' ').replace('\r', ' ')
            tokens = tokenize(content)
            out = []
            for token in tokens:
                out.append(tweetTextCleaner.tweetCleaner(token))
            data['text'] = ' '.join(out)
            cleanTweetFile.write(json.dumps(data)+'\n')
        tweetFile.close()
        cleanTweetFile.close()


if __name__ == '__main__':
    run()