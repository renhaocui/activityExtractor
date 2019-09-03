# activityExtractor
Extract user offline activities from their tweets.
Paper link: https://arxiv.org/pdf/1908.02551.pdf

The collector folder contains all the functions to collect the data, from places to tweets.

dataLabler actually just maps the labels to tweets and creates consolidate data files.
'processHistAttLSTM_contextPOST' in runHybrid is the main process on the proposed model.
trainFullModel trains and saves an activity recognition model on a given data.
'activityProfiling' utilize a trained model to infer activities on the tweets of the followers of given brands.
