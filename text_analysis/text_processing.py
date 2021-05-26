import os

import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk import re
from text_analysis.read_comments import read_comments_from_files
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

from wordcloud import WordCloud
from nltk import FreqDist
from nltk import bigrams
from operator import itemgetter

from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

import pyLDAvis
import pyLDAvis.gensim
import matplotlib.pyplot as plt

# for LDA
import gensim
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
from gensim import models

# for language detection
import fasttext

path_to_pretrained_model = 'C:\\Users\\Sofia\\PycharmProjects\\FishingTourismNLP\\text_analysis\\libs\\lid.176.bin'
language_model = fasttext.load_model(path_to_pretrained_model)

# for visualization
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})

visualize_topics = True
user_profile_analysis = True


def replace_locations(text):
    regex = r"kefalonia|zante|zak(?:i|y)+nthos|rhodes|mykonos|kos|paros|poros|koroni|santorini|thassos|iraklitsa|" \
            r"lefkada|l(?:e|i)+mnos|kissamos|milos|samos|skiathos|faliraki"
    replace = "tour_location"

    return re.sub(regex, replace, text)


def replace_owner_names(text):
    regex = r"dimitr(?:iou|io|i)s*|\bef{1,2}i|nan(?:c|s)+y|giannis*|marin*(?:a|o)+s*|mano(?:li)*s*|mike|mic*halis*|" \
            r"niko(?:la)*s*|artemis*|alex(?:andro|i)+s*|tasos*|al(?:i|y)+ki|g(?:i|e)+org(?:i|o)+s|george|efthimia|" \
            r"mohammed|eva|antonia|nikk*i(?:ta)*s*|mustafa|anton(?:io|i)+s*|vill*y|k(?:y|i)+riakos*|vett*a|giorgaros*|" \
            r"anthi|(?:ch|x)+ristina|yannis*|dion(?:i|y|u|h|e)+s(?:i|y|u|h|e)+s*|thaleia|(?:ch|x)+ristos*|" \
            r"andria(?:nna)*|panagi(?:oti)*s|theopoula|popp*(?:i|y)+|vass*il(?:eio|i)+s*|asimina|fross*o|tina|andreas*|" \
            r"geronim(?:u|o)s*|gerr*y|babis*|areti|nick|savvas*|stergos*|sergios*"

    replace = 'owner_name'

    return re.sub(regex, replace, text)


def is_english(text):
    result = language_model.predict(text)
    label = result[0][0]
    if label == '__label__en':
        return True
    return False


def text_cleaning(text):
    stop = stopwords.words('english') + ["would", "could", "also", "one", "ha", "can't", "it's", "i've", "u", "it",
                                         "us", "we", "t", "s"]  # define stopwords list

    # cleaning
    if pd.isnull(text):
        return ""
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text)  # remove URLs
    text = text.lower()  # to lowercase
    text = ''.join([i for i in text if not i.isdigit()])  # remove digits
    # text = text.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)  # remove unicodes and emojis
    text = re.sub(r'(.)\1+', r'\1\1', text)
    unis_emojis_pattern = re.compile(pattern="["
                                             u"\U0001F600-\U0001F64F"  # emoticons
                                             u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                             u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                             u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                             "]+", flags=re.UNICODE)
    text = unis_emojis_pattern.sub(r' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)  # remove punctuations
    split_text = text.split()
    text = ' '.join(x for x in text.split() if x not in stop)  # remove stopwords
    return text


def tokenize(text):
    words = word_tokenize(text)
    return words


def tokenize_and_stem(text):
    # first tokenize
    tokens = [word for word in nltk.word_tokenize(text)]
    # stemming
    stemmer = SnowballStemmer("english")
    stems = [stemmer.stem(t) for t in tokens]
    return stems


def tokenize_and_lemma(text):
    # first tokenize
    tokens = [word for word in nltk.word_tokenize(text)]
    # lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in tokens]
    return lemmas


def dummy_fun(doc):
    """ dummy function required for ngrams and tfidf tokenizer"""
    return doc


def ngrams(words, l1, l2):
    """
    words: words for which we create the vector
    l1: lower limit for ngrams
    l2: upper limit for ngrams
    Returns the vectors and the vocaabulary
    """
    cv = CountVectorizer(ngram_range=(l1, l2), tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=r'\b\w+\b',
                         min_df=1)
    ngrams = cv.fit_transform(words)
    return ngrams, cv


def tfidf(text):
    vect = TfidfVectorizer(max_df=0.8, max_features=200000,
                           min_df=0.2, stop_words='english',
                           use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 2))
    tfidf = vect.fit_transform(text)
    return tfidf, vect


def word_cloud(text, output_filepath):
    # Creating wordclouds
    cloud = WordCloud(width=800, height=400, background_color='white', collocations=False, colormap='Set2',
                      max_words=50).generate(' '.join(text))
    cloud.to_file(os.path.join(output_filepath, 'wordcloud.png'))


def ngrams_cloud(text, output_filepath):
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
                          colormap='Set2').generate_from_frequencies(
        word_dict)  # height=WC_height, width=WC_width, background_color='white')
    wordcloud.to_file(os.path.join(output_filepath, "bigrams_wordcloud.png"))


def word_frequencies_graph(text, output_filepath):
    text = ' '.join(text)
    # Calculate word frequencies
    words = word_tokenize(text)
    WNL = nltk.WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemmas.append(WNL.lemmatize(word))
    word_dist = FreqDist(lemmas)

    #
    # plt.bar(x, y, color=my_cmap(rescale(y)))
    # plt.savefig("temp")

    plt.style.use('fivethirtyeight')
    plt.tight_layout()

    rslt = pd.DataFrame(word_dist.most_common(20), columns=['Word', 'Frequency']).set_index('Word')
    fig = rslt.plot.bar(figsize=(15, 6)).get_figure()
    fig.savefig(os.path.join(output_filepath, 'word_frequencies_bar.png'), dpi=600)


def kmeans_topics(number_of_clusters, tfidf, df):
    # fitting kmeans
    print("Running k-means with K = " + str(number_of_clusters))
    num_clusters = number_of_clusters
    km = KMeans(n_clusters=num_clusters)
    km.fit(tfidf)

    # assign cluster number to each doc
    clusters = km.labels_.tolist()
    df['cluster'] = clusters
    # df['cluster'].value_counts()

    # creating a df from vocabulary
    totalvocab_stemmed = []
    totalvocab_tokenized = []
    for i in df['text_p']:
        allwords_stemmed = tokenize_and_stem(i)  # for each item in 'synopses', tokenize/stem
        totalvocab_stemmed.extend(allwords_stemmed)  # extend the 'totalvocab_stemmed' list
        allwords_tokenized = tokenize(i)
        totalvocab_tokenized.extend(allwords_tokenized)

    vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index=totalvocab_stemmed)

    # print the clusters and top words per cluster
    # print("Top terms per cluster:")
    # print()
    # sort cluster centers by proximity to centroid
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    # todo: Fix bug; cannot run with variable terms (unresolved)
    # for i in range(num_clusters):
    #     print("Cluster %d words:" % i, end='')
    #     for ind in order_centroids[i, :6]:  # replace 6 with n words per cluster
    #         print(' %s' % vocab_frame.loc[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'),
    #               end=',')
    #     print()  # add whitespace
    #     print()  # add whitespace

    return km, vocab_frame, df


# def lda_topics(text, number_of_topics, number_words):
#     def print_topics(model, count_vectorizer, n_top_words):
#         words = count_vectorizer.get_feature_names()
#         for topic_idx, topic in enumerate(model.components_):
#             print("\nTopic #%d:" % topic_idx)
#             print(" ".join([words[i]
#                             for i in topic.argsort()[:-n_top_words - 1:-1]]))
#
#     # Tweak the two parameters below
#     number_topics = number_of_topics
#     number_words = number_words
#     # Create and fit the LDA model
#
#     # Initialise the count vectorizer with the English stop words
#     count_vectorizer = CountVectorizer()
#     # Fit and transform the processed titles
#     count_data = count_vectorizer.fit_transform(text)
#     vocabulary = count_vectorizer.vocabulary_
#     lda = LDA(n_components=number_topics, n_jobs=-1)
#     lda.fit(count_data)
#
#     # Print the topics found by the LDA model
#     # Initialise the count vectorizer with the English stop words
#     print("Topics found via LDA:")
#     print_topics(lda, count_vectorizer, number_words)
#
#     return lda, count_vectorizer, count_data, vocabulary


def lda_topics(text, output_filepath):  # text-tokens
    """
    This function performs LDA analysis on the combination of Twitter and Instagram posts and extracts topics
    Simultaneously, it also visualises the topics based on the PyLDAVis library and saves the result as an HTML file
    :return:
    """
    global visualize_topics
    texts = []

    for tokens in text:
        texts.append(tokens)

    num_topics = 4

    texts_tokenized = [t.split() for t in texts]
    dictionary = gensim.corpora.Dictionary(texts_tokenized)

    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts_tokenized]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)

    topics = []

    for idx, topic in lda_model.print_topics(-1):

        topic_dict = dict()
        topic_dict["topic_id"] = idx
        topic_dict["topic_name"] = "topic_" + str(idx)
        topic_dict["significance"] = num_topics - idx

        words = topic.split(" + ")
        topic_terms = []

        for word in words:
            parts = word.replace("\"", "").split("*")
            weight = float(parts[0])
            term = parts[1]

            item = {"term": term, "weight": weight}
            topic_terms.append(item)

        topic_dict["terms"] = topic_terms
        topics.append(topic_dict)

    if visualize_topics:
        # pyLDAvis.enable_notebook()
        vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        pyLDAvis.save_html(vis, os.path.join(output_filepath, 'lda2_profiles.html'))

    return topics


def find_optimal_k_silhouette(df, tfidf, from_k=2, to_k=10):
    # defining the best k -- silhpuette score
    sil = []
    K = range(from_k, to_k)

    # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    for k in K:
        km, vocab_frame, df = kmeans_topics(k, tfidf, df)
        labels = km.labels_
        sil.append(silhouette_score(tfidf, labels, metric='euclidean'))

    # plot the distances
    plt.plot(K, sil, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette score')
    plt.title('Silhouette score For Optimal k')
    # plt.show()
    plt.savefig(os.path.join(output_filepath, 'silhouette.png'), dpi=600)


def find_optimal_k_elbow(df, tfidf, from_k=2, to_k=10):
    sum_of_squared_distances = []
    K = range(from_k, to_k)
    for k in K:
        km, vocab_frame, df = kmeans_topics(k, tfidf, df)
        sum_of_squared_distances.append(km.inertia_)

    # plot the distances
    plt.plot(K, sum_of_squared_distances, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Sum_of_squared_distances')
    plt.title('Elbow Method For Optimal k')
    plt.savefig(os.path.join(output_filepath, 'elbow.png'), dpi=600)


def setup_variables():
    if user_profile_analysis:
        filepath = 'data_collection\\output_profiles'
        text_column = 'review_details'
        title_column = 'reviewTitle'
        output_filepath = 'output\\output_user_profiles'
        processed_filepath = os.path.join(output_filepath, 'processed_dataframe.csv')
    else:
        filepath = 'data_collection\\output_reviews'
        text_column = 'text'
        title_column = 'title'
        output_filepath = 'output\\output_business_profiles'
        processed_filepath = os.path.join(output_filepath, 'processed_dataframe.csv')
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)
    return filepath, text_column, title_column, output_filepath, processed_filepath


if __name__ == '__main__':
    filepath, text_column, title_column, output_filepath, processed_filepath = setup_variables()
    if os.path.exists(processed_filepath):
        df = pd.read_csv(processed_filepath)
    else:
        df = read_comments_from_files(input=filepath, user_profiles=user_profile_analysis)

        # Filter out entries without text (comment field is nan)
        print('Filtering out empty-text reviews..')
        df.dropna(subset=[text_column], inplace=True)

        # Text cleaning
        print("Cleaning text...")
        df['text_p'] = df[text_column].apply(lambda x: text_cleaning(x))  # for user profiles

        # Filter out non-English entries
        print('Filtering out non-English reviews...')
        df = df[df['text_p'].map(lambda x: is_english(x))]

        # Replacing names and tour locations
        print("Replacing owner and crew's names and tour locations...")
        df['text_p'] = df['text_p'].apply(lambda x: replace_owner_names(x))
        df['text_p'] = df['text_p'].apply(lambda x: replace_locations(x))

        # Tokenization, Lemmatization and Stemming
        print("Tokenization, Lemmatization and Stemming...")
        df['text_w'] = df['text_p'].apply(lambda x: tokenize(x))
        df['text_l'] = df['text_p'].apply(lambda x: tokenize_and_lemma(x))
        df['text_s'] = df['text_p'].apply(lambda x: tokenize_and_stem(x))
        # df['title_p'] = df['title'].apply(lambda x: text_cleaning(x))  # for business reviews
        df['title_p'] = df[title_column].apply(lambda x: text_cleaning(x))  # for user profiles
        df['title_w'] = df['title_p'].apply(lambda x: tokenize(x))
        df['title_l'] = df['title_p'].apply(lambda x: tokenize_and_lemma(x))
        df['title_s'] = df['text_p'].apply(lambda x: tokenize_and_stem(x))

    # Ngram and tfidf Vectors
    print("N-grams and TF-IDF Vectors...")
    ngrams, ngram_voc = ngrams(df['text_l'], 2, 4)  # returns bigrams and trigrams
    tfidf, tfidf_voc = tfidf(df['text_p'])  # returns unigrams

    # topics extraction
    # Identifying the optimal number of clusters
    # print("Identifying the optimal number of clusters...")
    # find_optimal_k_silhouette(df, tfidf, from_k=2, to_k=20)

    # with LDA (w/o Gensim)
    # number_of_topics = _INSERT_
    # number_words = _INSERT_
    # lda, count_vectorizer, count_data, vocab = lda_topics(df['text_p'], number_of_topics, number_words)
    # Visualize lda
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(lda, count_data, vocab)
    # vis

    # with LDA (w/ Gensim)
    print("LDA...")
    topics = lda_topics(df['text_p'], output_filepath)

    # # with kmeans
    # # defining the best k -- elbow method
    # number_of_clusters = _INSERT_OPTIMAL_
    # km, vocab_frame, df = kmeans_topics(number_of_clusters, tfidf, df)

    # Vizualization and stats - word clouds, word bars etc
    word_cloud(df['text_p'], output_filepath)
    ngrams_cloud(df['text_p'], output_filepath)
    word_frequencies_graph(df['text_p'], output_filepath)

    df.to_csv(os.path.join(output_filepath,'processed_dataframe.csv'))
