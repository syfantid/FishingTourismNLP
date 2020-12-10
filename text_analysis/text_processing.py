import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk import re
from text_analysis.read_comments import read_comments_from_files
from nltk.tokenize import word_tokenize, RegexpTokenizer


from wordcloud import WordCloud
from nltk import FreqDist
from nltk import bigrams
from operator import itemgetter


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


def word_cloud(text):
    # Creating wordclouds
    cloud = WordCloud(width=800, height=400, background_color='white', collocations=False, colormap='Set2',
                        max_words=50).generate(' '.join(text))
    cloud.to_file('wordcloud.png')

def ngrams_cloud(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = ' '.join(text)
    sent_words = tokenizer.tokenize(text)

    # Calculate the frequency distance
    freq_dist = FreqDist(bigrams(sent_words))
    # Sort highest to lowest based on the score.
    scoredList = sorted(freq_dist.items(), key=itemgetter(1), reverse=True)
    # word_dict is the dictionary we'll use for the word cloud.
    # Load dictionary with the FOR loop below.
    word_dict = {}
    # Get the bigram and make a contiguous string for the dictionary key.
    # Set the key to the scored value.
    listLen = len(scoredList)
    for i in range(listLen):
        word_dict['_'.join(scoredList[i][0])] = scoredList[i][1]

    WC_max_words = 50
    wordcloud = WordCloud(max_words=WC_max_words, height=400, width=800, collocations=False, background_color='white',
                          colormap='Set2').generate_from_frequencies(word_dict)  # height=WC_height, width=WC_width, background_color='white')
    wordcloud.to_file("bigrams_wordcloud.png")

def word_frequencies_graph(text):

    text = ' '.join(text)
    #Calculate word frequencies
    words = word_tokenize(text)
    WNL = nltk.WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemmas.append(WNL.lemmatize(word))
    word_dist = FreqDist(lemmas)
    rslt = pd.DataFrame(word_dist.most_common(20), columns=['Word', 'Frequency']).set_index('Word')
    fig = rslt.plot.bar(rot=45).get_figure()
    fig.savefig('word_frequencies_bar.png')



# df = read_comments_from_files()
# df['text_p'] = df['text'].apply(lambda x: text_cleaning(x))
# df['text_w'] = df['text_p'].apply(lambda x: text_token(x))
# df['title_p'] = df['title'].apply(lambda x: text_cleaning(x))
# df['title_w'] = df['title_p'].apply(lambda x: text_token(x))

# Vizualization and stats - word clouds, word bars etc
# word_cloud(df['text_p'])
# ngrams_cloud(df['text_p'])
# word_frequencies_graph(df['text_p'])

#todo Dimitra! topic extraction LDA
#todo Dimitra! words-ngrams coocurrence network visualization

