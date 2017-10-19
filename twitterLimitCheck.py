import time
__author__ = 'renhao.cui'
import json
import twitter

c_k = 'Njg6fcGHob5BqQ6EIckIJf7vd'
c_s = 'u6R1ktMNgIslBcQifkLqrNKjNoelYviE1rWU0CIqcmRlnqWir0'
a_t = '141612471-TrScxwz2F6Bc3rqfwexHZMP6Ri52zAADyagS1UIz'
a_t_s = '9YnjV3VzGZEFOgdKxqEgDVt1GnxVvMX8V1m47EN2j71fl'


def oauth_login():
    # credentials for OAuth
    CONSUMER_KEY = c_k
    CONSUMER_SECRET = c_s
    OAUTH_TOKEN = a_t
    OAUTH_TOKEN_SECRET = a_t_s
    # Creating the authentification
    auth = twitter.oauth.OAuth(OAUTH_TOKEN,
                               OAUTH_TOKEN_SECRET,
                               CONSUMER_KEY,
                               CONSUMER_SECRET)
    # Twitter instance
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api

def check():
    twitter_api = oauth_login()
    response = twitter_api.application.rate_limit_status(resources='search,statuses')
    print response

if __name__ == '__main__':
    check()