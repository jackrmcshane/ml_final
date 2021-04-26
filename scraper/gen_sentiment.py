import flair
import pandas as pd
from the_pickler import *






if __name__ == '__main__':


    """"""
    csv_filename = '../data/tweets.csv'
    df = pd.read_csv(csv_filename)

    tweets = list(df.tweets)

    """"""


    # for each tweet gen sentiment
    sentiment_model = flair.models.TextClassifier.load('en-sentiment')
    tweet = tweets[0]
    sentence = flair.data.Sentence(tweet)
    print(sentence)
    sentiment_model.predict(sentence)
    #sentiments = [sentiment_model.predict(tweet).labels[0].value for tweet in tweets]
    #print(pd.DataFrame(sentiments).info())


    missed = list()
    sentiments = list()
    for i, tweet in enumerate(tweets):
        try:
            # create sentence object
            sent = flair.data.Sentence(tweet)
            sentiment_model.predict(sent)
            sentiments.append(sent.labels[0].value)
        except:
            missed.append(i)
            print('An error has occurred. The sentence responsible is at this index in tweets: ', i)
            print('Here is the tweet: ', tweet)
            print('len of the tweet: ', len(tweet))
            print('-----------------------------------------------------------------------------')






    print('final len of the sentiments list: ', len(sentiments))

    if missed:
        print('number of missed tweets: {}'.format(168000 - len(sentiments)))
        print('the indices of the missed tweets are: ', missed)
        for ind in missed:
            del tweets[ind]



    path_to_csv = '../data/df.csv'
    df = pd.DataFrame(list(zip(tweets, sentiments)), columns=['tweets', 'sentiments'])
    df.to_csv(path_to_csv)




    #print(pd.DataFrame(sentiments).head())
    #print(pd.DataFrame.value_counts())
    #df = pd.DataFrame(sentiments)
    #df.to_csv('sentiments.csv', sep=',')
