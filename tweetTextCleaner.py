import re

__author__ = 'rencui'

charList = ['&gt;', '&amp;', '|', '&lt;3', '()', 'amp']

def removeEmoji(input, token):
    emojis = re.findall(r'\\u....', input)
    if len(emojis) != 0:
        for char in emojis:
            input = input.replace(char, token)
    return input

def shrinkPuncuation(input):
    input = re.sub('\.+', '.', input)
    input = re.sub(',+', ',', input)
    input = re.sub(' +', ' ', input)
    input = re.sub('=+', '=', input)
    input = re.sub('-+', '-', input)
    input = re.sub('_+', '_', input)
    input = re.sub(' +', ' ', input)
    input = re.sub('\s+', ' ', input)
    return input

def removeUsername(input, token):
    users = re.findall(r'@(\w+)', input)
    if len(users) != 0:
        for user in users:
            input = input.replace(user, token)
            input = input.replace('@', '')
    return input

def tokenizeLinks(input, token):
    urls = re.findall("(?P<url>https?://[^\s]+)", input)
    if len(urls) != 0:
        for url in urls:
            input = input.replace(url, token)
    return input

def removeHashtag(input, token):
    hts = re.findall(r'#(\w+)', input)
    if len(hts) != 0:
        for ht in hts:
            input = input.replace(ht, token)
            input = input.replace('#', '')
    return input

def removeRT(input):
    if input.startswith('RT : '):
        return True
    else:
        return False

def tweetCleaner(input):
    input = input.replace('w/', 'with')
    input = input.replace('w/o', 'without')
    input = removeUsername(input, '')
    input = removeHashtag(input, '')
    input = removeEmoji(input, '')
    input = tokenizeLinks(input, '')
    for char in charList:
        input = input.replace(char, '')
    input = input.replace('\\"', '"')
    input = input.replace('\\', '')
    input = shrinkPuncuation(input)
    if input != '':
        if input[0] == ' ':
            input = input[1:]
    return input
