import json

def yelpDataGenerate(reviewFilename, businessFilename, outputFilename):
    businessData = {}
    businessFile = open(businessFilename, 'r')
    for line in businessFile:
        item = json.loads(line.strip())
        if item['business_id'] not in businessData:
            businessData[item['business_id']] = item['stars']
    businessFile.close()
    print len(businessData)

    outputFile = open(outputFilename, 'w')
    reviewFile = open(reviewFilename, 'r')
    outputData = {}
    lineCount = 0
    writtenBusinessSet = set()
    for line in reviewFile:
        item = json.loads(line.strip())
        businessID = item['business_id']
        if businessID in businessData:
            content = item['text'].replace('\n', ' ').replace('\r', ' ')
            if businessID in outputData:
                if len(outputData[businessID]['reviews']) == 20 and businessID not in writtenBusinessSet:
                    outputFile.write(json.dumps(outputData[businessID])+'\n')
                    writtenBusinessSet.add(businessID)
                    lineCount += 1
                elif len(outputData[businessID]['reviews']) < 20:
                    outputData[businessID]['reviews'].append({'review_id': item['review_id'], 'user_id': item['user_id'], 'stars': item['stars'], 'date': item['date'], 'text': content})
            else:
                outputData[businessID] = {'business_id': businessID, 'business_stars': businessData[businessID], 'reviews': [{'review_id': item['review_id'], 'user_id': item['user_id'], 'review_stars': item['stars'], 'date': item['date'], 'text': content}]}
    reviewFile.close()
    outputFile.close()
    print lineCount


def yelpDataGenerator2(reviewFilename, outputFilename):
    outputFile = open(outputFilename, 'w')
    reviewFile = open(reviewFilename, 'r')
    userReviewData = {}
    lineCount = 0
    writtenUserSet = set()
    for line in reviewFile:
        data = json.loads(line.strip())
        userID = data['user_id']
        content = data['text'].replace('\n', ' ').replace('\r', ' ')
        if userID in userReviewData:
            if len(userReviewData[userID]['reviews']) == 10 and userID not in writtenUserSet:
                outputFile.write(json.dumps(userReviewData[userID])+'\n')
                writtenUserSet.add(userID)
                lineCount += 1
            elif len(userReviewData[userID]['reviews']) < 10:
                userReviewData[userID]['reviews'].append({'review_id': data['review_id'], 'business_id': data['business_id'], 'stars': data['stars'], 'date': data['date'], 'text': content})
        else:
            userReviewData[userID] = {'user_id': userID, 'reviews': [{'review_id': data['review_id'], 'business_id': data['business_id'], 'stars': data['stars'], 'date': data['date'], 'text': content}]}
    reviewFile.close()
    outputFile.close()
    print lineCount


if __name__ == '__main__':
    #yelpDataGenerate('data/yelp_dataset/review.json', 'data/yelp_dataset/business.json', 'data/yelpData.json')
    yelpDataGenerator2('data/yelp_dataset/review.json', 'data/yelpUserReviewData.json')