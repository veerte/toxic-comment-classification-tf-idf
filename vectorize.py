
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from cleantext import clean
import contractions


from reader import get_comments

print("loading nltk")
import nltk
nltk.download('punkt')

print("loading data")
x_train, y_train, x_test, y_test = get_comments()
x = x_train

def cleanfunc(x):
    return clean(contractions.fix(x),
    fix_unicode=True,               # fix various unicode errors
    to_ascii=True,                  # transliterate to closest ASCII representation
    lower=True,                     # lowercase text
    no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
    no_urls=True,                  # replace all URLs with a special token
    no_emails=True,                # replace all email addresses with a special token
    no_phone_numbers=True,         # replace all phone numbers with a special token
    no_numbers=True,               # replace all numbers with a special token
    no_digits=True,                # replace all digits with a special token
    no_currency_symbols=True,      # replace all currency symbols with a special token
    no_punct=False,                 # remove punctuations
    replace_with_punct="",          # instead of removing punctuations you may replace them
    replace_with_url="",
    replace_with_email="",
    replace_with_phone_number="",
    replace_with_number="",
    replace_with_digit="",
    replace_with_currency_symbol="",
    lang="en")

print("cleaning text")

with ProcessPoolExecutor() as ppe:
    x_cleaned = list(ppe.map(cleanfunc, x))

import re
import itertools

stemmer = nltk.stem.snowball.SnowballStemmer("english")

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    st = nltk.sent_tokenize(text)
    wt = map(nltk.word_tokenize, st)
    tokens = itertools.chain.from_iterable(wt)

    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    filtered_tokens = filter(lambda t: re.search('[a-zA-Z]', t), tokens)
    stems = map(stemmer.stem, filtered_tokens)
    return ' '.join(list(stems))

print("stemming")

with ProcessPoolExecutor() as ppe:
    x_stemmed = list(ppe.map(tokenize_and_stem, x_cleaned))


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')

print("fitting vectorizer")

vectorizer.fit(x_stemmed)
import pickle
with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)

def vectorize(t):
    return vectorizer.transform([t])

print("vectorizing stemmed comments")

with ProcessPoolExecutor() as ppe:
    results = list(ppe.map(vectorize, x_stemmed))

with open('results.pickle', 'wb') as f:
    pickle.dump(results, f)

print("done")