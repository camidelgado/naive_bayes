from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
#convertir texto en un valor a una matrix, cada columna es una palabra
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import unicodedata
import pickle
import re
from sklearn.metrics import classification_report
from sklearn import metrics

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')
df['review']=df['review'].str.strip()
df['review']=df['review'].str.lower()
df['review']=df['review'].str.replace('!','')
df['review']=df['review'].str.replace(',','')
df['review']=df['review'].str.normalize('NFKC') 
df['review']=df['review'].str.replace(r'([a-zA-Z])\1{2,}',r'\1',regex=True)

def normalize_string(text_string):
    if text_string is not None:
        result=unicodedata.normalize('NFD',text_string).encode('ascii','ignore').decode()
    else: 
        result=None
    return result

df['review']=df['review'].apply(normalize_string)

X=df['review']
y=df['polarity']

X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=563, stratify=y)text_clf = Pipeline([('vec',CountVectorizer(stop_words='english')),('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf.fit(X_train, y_train)
vec.get_feature_names_out()
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

y_pred = text_clf.predict(X_test)
n_iter_search = 5
parameters = {'vec__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
gs_clf = RandomizedSearchCV(text_clf, parameters, n_iter = n_iter_search)
gs_clf.fit(X_train, y_train)

y_pred_mejor = gs_clf.predict(X_test)

pickle.dump(best_model, open('../models/best_model.pickle', 'wb'))

