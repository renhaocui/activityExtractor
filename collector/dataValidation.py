import json, os

def extractor():
    tweetIDSet = set()
    for file in os.listdir("data/google_places3"):
        placeName = file.split('.')[0]
        print 'Extracting: ' + placeName
        tweetFile = open('data/google_place_tweets3.3_out/'+placeName+'.json', 'r')
        outFile = open('data/POIplace/'+placeName+'.json', 'w')

        for line in tweetFile:
            if line.strip().endswith('}'):
                try:
                    tweet = json.loads(line.strip())
                except:
                    continue
                if tweet['place'] is not None:
                    if tweet['place']['place_type'] == 'poi':
                        if tweet['id'] not in tweetIDSet:
                            tweetIDSet.add(tweet['id'])
                            outFile.write(line)

        tweetFile.close()
        outFile.close()


if __name__ == '__main__':
    extractor()