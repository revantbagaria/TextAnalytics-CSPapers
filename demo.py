import re, nltk, string, sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


CORPUS = [
'the sky is blue',
'sky is blue and sky is beautiful',
'the beautiful sky is so blue',
'i love blue cheese'
]
new_doc = ['loving this blue sky today']


vectorizer = CountVectorizer(min_df = 1, ngram_range = (1,1))
features = vectorizer.fit_transform(CORPUS)
# print(vectorizer.transform(CORPUS))
print(vectorizer.get_feature_names())
print(features)
