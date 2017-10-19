import os

for file in os.listdir("data/google_place_tweets3.3_new"):
    oldFile = open('data/google_place_tweets3.3/' + file, 'r')
    newFile = open('data/google_place_tweets3.3_new/' + file, 'r')
    outFile = open('data/google_place_tweets3.3_out/' + file, 'w')

    for line in oldFile:
        outFile.write(line)
    for line in newFile:
        outFile.write(line)

    oldFile.close()
    newFile.close()
    outFile.close()