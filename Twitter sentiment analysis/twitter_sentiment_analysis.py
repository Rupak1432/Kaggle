#import nltk
#nltk.download()

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from nltk.stem.snowball import SnowballStemmer

data = pd.read_csv("train.csv", encoding='latin-1')
dict_of_lists = {}

if __name__ == '__main__':
    for column_name in data.columns:
        temp = data[column_name].tolist()
        dict_of_lists[column_name] = temp

    x_train = dict_of_lists['SentimentText'][1:80001]
    x_test = dict_of_lists['SentimentText'][80001:]

    y_train = dict_of_lists['Sentiment'][1:80001]
    y_test = dict_of_lists['Sentiment'][80001:]

    def clean_tweet(w):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", w).split())

    for i in range(len(x_train)):
        x_train[i] = clean_tweet(x_train[i])

    for i in range(len(x_test)):
        x_test[i] = clean_tweet(x_test[i])


    # Naive Bayes
    twit_clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultinomialNB()),
                        ])

    twit_clf = twit_clf.fit(x_train,y_train)
    predicted = twit_clf.predict(x_test)
    print('naive bayes: ',np.mean(predicted == y_test))

    # SVM
    twit_clf_svm = Pipeline([('vect',CountVectorizer(stop_words = 'english')),
                            ('tfidf',TfidfTransformer()),
                            ('clf-svm',SGDClassifier(loss = 'hinge', penalty = 'l2',
                                                     alpha = 1e-3, n_iter = 5,
                                                     random_state = 42)),
                            ])

    twit_clf_svm = twit_clf_svm.fit(x_train,y_train)
    predicted_svm = twit_clf_svm.predict(x_test)
    print('svm: ',np.mean(predicted_svm == y_test))


    ## Grid Search CV

    # naive bayes
    parameters = {'vect__ngram_range':[(1,1),(1,2)],
                 'tfidf__use_idf':(True,False),
                 'clf__alpha': (1e-2, 1e-3),
                 }

    gs_clf = GridSearchCV(twit_clf, parameters, n_jobs = -1)
    gs_clf = gs_clf.fit(x_train, y_train)
    print('gs_clf - best score',gs_clf.best_score_)
    print('gs_clf - best params',gs_clf.best_params_)
    predicted_gs = gs_clf.predict(x_test)
    print('naive bayes gs: ',np.mean(predicted_gs == y_test))

    # svm
    parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)],
                       'tfidf__use_idf': (True, False),
                       'clf-svm__alpha': (1e-2, 1e-3),
     }
    gs_clf_svm = GridSearchCV(twit_clf_svm, parameters_svm, n_jobs=-1)
    gs_clf_svm = gs_clf_svm.fit(x_train, y_train)
    print('gs_clf_svm.best_score: ',gs_clf_svm.best_score_)
    print('gs_clf_svm.best_params: ',gs_clf_svm.best_params_)
    predicted_gs_svm = gs_clf_svm.predict(x_test)
    print('svm gs: ',np.mean(predicted_gs_svm == y_test))


    ## stemming

    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    class StemmedCountVectorizer(CountVectorizer):
        def build_analyzer(self):
            analyzer = super(StemmedCountVectorizer, self).build_analyzer()
            return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')


    # Naive Bayes
    twit_mnb_stemmed = Pipeline([('vect', stemmed_count_vect),
                                 ('tfidf', TfidfTransformer()),
                                 ('mnb', MultinomialNB(fit_prior=False)),
     ])

    twit_mnb_stemmed = twit_mnb_stemmed.fit(x_train, y_train)
    predicted_mnb_stemmed = twit_mnb_stemmed.predict(x_test)
    print('naive bayes with stemming: ',np.mean(predicted_mnb_stemmed == y_test))

    #svm
    twit_svm_stemmed = Pipeline([('vect', stemmed_count_vect),
                          ('tfidf', TfidfTransformer()),
                          ('svm', SGDClassifier(loss = 'hinge', penalty = 'l2',
                                                     alpha = 1e-3, n_iter = 5,
                                                     random_state = 42)),
     ])

    twit_svm_stemmed = twit_svm_stemmed.fit(x_train, y_train)
    predicted_svm_stemmed = twit_svm_stemmed.predict(x_test)
    print('svm with stemming: ',np.mean(predicted_svm_stemmed == y_test))

    