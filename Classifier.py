import numpy as np
import twitter
import re, datetime, pandas as pd
from textacy.preprocess import preprocess_text 
import ftfy
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


#TweetMiner function from Mike Roman

class TweetMiner(object): 
 
    
    def __init__(self, api, result_limit = 20):
        
        self.api = api        
        self.result_limit = result_limit
        

    def mine_user_tweets(self, user="HillaryClinton", mine_retweets=False, max_pages=40):

        data           =  []
        last_tweet_id  =  False
        page           =  1
        
        while page <= max_pages:
            
            if last_tweet_id:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, max_id=last_tweet_id - 1, include_rts=mine_retweets)
                statuses = [ _.AsDict() for _ in statuses]
            else:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, include_rts=mine_retweets)
                statuses = [_.AsDict() for _ in statuses]
                
            for item in statuses:
                # Using try except here.
                # When retweets = 0 we get an error (GetUserTimeline fails to create a key, 'retweet_count')
                try:
                    mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   item['retweet_count'],
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                except:
                        mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   0,
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    } 
                
                last_tweet_id = item['id']
                data.append(mined)
                
            page += 1
            
        return data


twitter_keys = {
    'consumer_key':        'L4sziHBqV4VUIfKezbos0JMVl',
    'consumer_secret':     'lJau6R7GIHFwoGR5wB3PlLQPXBChwzJFJ9WGXXtazcDSA1Vb1X',
    'access_token_key':    '941359629606539264-05XcmQfdwMXTbPNWS3r7cZThvbQBxCK',
    'access_token_secret': 'VdE3VJVk6oxbohQGcw7WYA5Tg4Sr8kW9duTO1wxmB6qXk'
}

api = twitter.Api(
    consumer_key         =   twitter_keys['consumer_key'],
    consumer_secret      =   twitter_keys['consumer_secret'],
    access_token_key     =   twitter_keys['access_token_key'],
    access_token_secret  =   twitter_keys['access_token_secret'],
    tweet_mode = 'extended' 
)



#retreving user tweets 
miner = TweetMiner(api, result_limit=200)
trump_tweets = miner.mine_user_tweets("realDonaldTrump")

trump_df = pd.DataFrame(trump_tweets) 

hillary_tweets = miner.mine_user_tweets('HillaryClinton') 

hillary_df = pd.DataFrame(hillary_tweets)

tweets = pd.concat([trump_df, hillary_df], axis=0)



tweet_text = tweets['text'].values
clean_text = [preprocess_text(x, fix_unicode=True, lowercase=True, no_urls=True, no_emails=True, no_phone_numbers=True, no_currency_symbols=True,no_punct=True, no_accents=True)
                for x in tweet_text]

y =tweets['handle'].map(lambda x: 1 if x == 'realDonaldTrump' else 0).values
print (max(pd.Series(y).value_counts(normalize=True)))


#Vectorizing with TF-IDF Vectorizer and creating X matrix
tfv = TfidfVectorizer(min_df=1,ngram_range=(1,4),max_features=2500)
X = tfv.fit_transform(clean_text).todense()
print (X.shape)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

lr = LogisticRegression(solver='liblinear')
params = {'penalty': ['l1', 'l2'], 'C':np.logspace(-5,0,100)}
#Grid searching to find optimal parameters for Logistic Regression
gs = GridSearchCV(lr, param_grid=params, cv=4, verbose=1)
#gs=RandomizedSearchCV(lr,params)
gs.fit(X_train, y_train)

print (gs.best_params_)
print (gs.score(X_train,y=y_train))
print (gs.score(X_test,y=y_test))



estimator = LogisticRegression(solver='liblinear',penalty='l2',C=1.0)
estimator.fit(X,y)

# Prep our source as TfIdf vectors
source_test = [
    "The presidency doesn’t change who you are—it reveals who you are. And we’ve seen all we need to of Donald Trump.",
    "Crooked Hillary is spending tremendous amounts of Wall Street money on false ads against me. She is a very dishonest person!"
]


Xtest = tfv.transform(source_test)
print(pd.DataFrame(estimator.predict_proba(Xtest), columns=["Proba_Hillary", "Proba_Trump"]))