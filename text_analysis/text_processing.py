import os
import pandas as pd
import string
import nltk
import todo as todo
from nltk.corpus import stopwords
from nltk import re
from text_analysis.read_comments import read_comments_from_files
from nltk.tokenize import word_tokenize

def text_cleaning(text):
    stop = stopwords.words('english')  # define stopwords list

    # cleaning
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text) #remove URLs
    text = text.lower() # to lowercase
    text = ''.join([i for i in text if not i.isdigit()])  # remove digits
    text = ' '.join(x for x in text.split() if x not in stop)  # remove stopwords
    #text = text.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)  # remove unicodes and emojis
    unis_emojis_pattern = re.compile(pattern="["
                                        u"\U0001F600-\U0001F64F"  # emoticons
                                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                        "]+", flags=re.UNICODE)
    text = unis_emojis_pattern.sub(r' ', text)
    text = re.sub(r'[^\w\s]',' ',text) # remove punctuations
    return text

def text_token(text):
    words = word_tokenize(text)
    return words

# df = read_comments_from_files()
# df['text_p'] = df['text'].apply(lambda x: text_cleaning(x))
# df['text_w'] = df['text_p'].apply(lambda x: text_token(x))
# df['title_p'] = df['title'].apply(lambda x: text_cleaning(x))
# df['title_w'] = df['title_p'].apply(lambda x: text_token(x))


#todo Dimitra! Vizualization and stats - word clouds, word bars etc
#todo Dimitra! topic extraction LDA
#todo Dimitra! words-ngrams coocurrence network visualization

