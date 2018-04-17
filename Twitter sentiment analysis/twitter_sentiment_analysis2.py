import re
import tweepy
from textblob import TextBlob

Consumer_Key = ''  
Consumer_Secret = ''  

Access_Token = ''
Access_Token_Secret =  ''

auth = tweepy.OAuthHandler(Consumer_Key, Consumer_Secret)
auth.set_access_token(Access_Token,Access_Token_Secret)

api = tweepy.API(auth)

def clean_tweet(w):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", w).split())

def sentiment(w):
    analysis = TextBlob(clean_tweet(w))

    if analysis.sentiment.polarity > 0:
        return('Positive')
    elif analysis.sentiment.polarity == 0:
        return('Neutral')
    else:
        return('Negative')

def senti_analysis(w,n):
    analysed_tweet = []

    data = api.search(w,count = n)

    for i in data:
        tweet = {}
        tweet['string'] = i.text
        tweet['sentiment'] = sentiment(i.text)

        if i.retweet_count>0:
            if tweet not in analysed_tweet:
                analysed_tweet.append(tweet)
        else:
            analysed_tweet.append(tweet)

    return analysed_tweet

def main():
    phrase = input('Enter the phrase/word to be analysed: ')

    tweet_count = input('\nEnter the Number of tweets to be analysed: ')

    analysed_sentiment = senti_analysis(phrase,tweet_count)

    for i in analysed_sentiment:
        print('\n' + i['string'] + ' = ' + i['sentiment'])
    

if __name__ == '__main__':
    main()