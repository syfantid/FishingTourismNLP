import os
from pprint import pprint
from time import time
import random
import re

import pandas as pd
from matplotlib.pyplot import bar
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, plot_confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
import matplotlib.pyplot as plt

from text_analysis.read_comments import read_comments_from_files

from spellchecker import SpellChecker
from nltk import pos_tag
from nltk.corpus import names
from nltk import sent_tokenize
from nltk import NaiveBayesClassifier
import textstat
from text_analysis.sentiment_analysis import get_review_sentiment

from imblearn.over_sampling import SMOTE


INPUT_PATH = 'output\\output_user_profiles'
INPUT_FILENAME = 'processed_dataframe.csv'
DEMOGRAPHICS_PATH = 'data_collection\\output_demographics'
MODELS_PATH = '' #TODO add Sofia

# For mac
# INPUT_PATH = 'text_analysis/output/output_user_profiles'
# INPUT_FILENAME = 'processed_dataframe.csv'
# DEMOGRAPHICS_PATH = 'FishingTourismNLP/data_collection/output_demographics'
# MODELS_PATH = 'models'


def grid_search(pipeline, parameters, X_train, y_train, filename):
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    clf = grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print()

    with open(os.path.join('output', filename), 'w') as f:  # open the file
        f.write("Best score: %0.3f\n" % grid_search.best_score_)
        print("Best score: %0.3f" % grid_search.best_score_)
        f.write("Best parameters set:\n")
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
            f.write("\t%s: %r\n" % (param_name, best_parameters[param_name]))

    f.close()

    return clf


def evaluate(y_test, y_predicted, clf, X_test, model_name, X_train=None, y_train=None):
    print(model_name)
    if X_train is not None and y_train is not None:
        print("Evaluation in Training Set:")
        score = clf.score(X_train, y_train)
        print(str(score))
    print("Accuracy: " + str(mean(y_predicted == y_test)))
    print("F1-score macro: " + str(f1_score(y_test, y_predicted, average='weighted')))
    plot_confusion_matrix(clf, X_test, y_test)
    plt.show()


def fix_labels(df):
    df['gender'] = df['gender'].apply(lambda x: x.lower())
    return df



def nb(X_train, X_test, y_train, y_test):
    pipeline_nb = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])

    clf_nb = pipeline_nb.fit(X_train, y_train)
    y_predicted = clf_nb.predict(X_test)
    evaluate(y_test, y_predicted, clf_nb, X_test, "Multinomial Bernoulli")



def lr(X_train, X_test, y_train, y_test):
    pipeline_lr = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression())
    ])

    clf_lr = pipeline_lr.fit(X_train, y_train)
    y_predicted = clf_lr.predict(X_test)
    evaluate(y_test, y_predicted, clf_lr, X_test, "Logistic Regression")


def rf(X_train, X_test, y_train, y_test):
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer()),
        ('lr', RandomForestClassifier())
    ])

    clf_rf = pipeline_rf.fit(X_train, y_train)
    y_predicted = clf_rf.predict(X_test)
    evaluate(y_test, y_predicted, clf_rf, X_test, "Random Forest", X_train=X_train, y_train=y_train)


def sgd(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', SGDClassifier()),
    ])

    # uncommenting more parameters will give better exploring power but will
    # increase processing time in a combinatorial way
    parameters = {
        'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        # 'tfidf__use_idf': (True, False),
        # 'tfidf__norm': ('l1', 'l2'),
        'clf__max_iter': (20,),
        'clf__alpha': (0.00001, 0.000001),
        # 'clf__penalty': ('l2', 'elasticnet'),
        # 'clf__max_iter': (10, 50, 80),
    }

    clf_sgd = grid_search(pipeline, parameters, X_train, y_train, 'sgd.csv')
    y_predicted = clf_sgd.predict(X_test)
    evaluate(y_test, y_predicted, clf_sgd, X_test, "Stochastic Gradient Descent")


def gender_prediction(df, classifier='lr'):
    df.dropna(subset=['gender'], inplace=True)
    df = fix_labels(df)

    # Data to use
    # df = df[['text_p', 'title_p', 'reviewFor', 'reviewRating', 'gender']]
    X = df['text_p']
    y = df['gender']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=33)
    # gender_prediction_nb(X_train, X_test, y_train, y_test)
    # gender_prediction_lr(X_train, X_test, y_train, y_test)
    if classifier == 'nb':
        nb(X_train, X_test, y_train, y_test)
    elif classifier == 'lr':
        lr(X_train, X_test, y_train, y_test)
    elif classifier == 'rf':
        rf(X_train, X_test, y_train, y_test)
    else:
        sgd(X_train, X_test, y_train, y_test)
    # param_grid = [{'logisticregression__C': [1, 10, 100, 1000]}
    # gs = GridSearchCV(pipe, param_grid)
    # gs.fit(X, y)

def age_features(df):

    # structure features
    df['no_char'] = df['review_details'].str.len()  #number of characters
    df['no_words'] = df['review_details'].str.split().str.len() #number of words

    def average_words(text):
        counts = []
        sentences = sent_tokenize(text)
        for sentence in sentences:
            counts.append(sentence.split())
        words = sum([len(element) for element in counts])
        return float(words)/len(counts)

    df['sentence_words'] = df['review_details'].apply(lambda x: average_words(x)) #avg number of words per sentence
    df['exclamatories'] = df['review_details'].str.count('!') #number of exclamatories
    #df['no_misspelled'] = df['review_details'].str.split().apply(lambda x: len(list(SpellChecker.unknown(x)))) #number of misspelled words

    # syntax
    df['pos_tag'] = df['text_p'].apply(lambda x: pos_tag(x.split(" ")))
    df['tags'] = df['pos_tag'].apply(lambda x: [pos for word, pos in (x)])
    # # summing to larger POS groups
    df['Adj'] = df['tags'].apply(lambda x: x.count('JJ') + x.count('JJR') + x.count('JJS'))
    df['Verb'] = df['tags'].apply(lambda x: x.count('VB') + x.count('VBD') + x.count('VBG') + x.count('VBN') + x.count('VBP') + x.count('VBZ'))
    df['Noun'] = df['tags'].apply(lambda x: x.count('NN') + x.count('NNS') + x.count('NNP') + x.count('NNPS'))
    df['Adv'] = df['tags'].apply(lambda x: x.count('RB') + x.count('RBR') + x.count('RBS'))


    # Readabilty
    df['flesch_reading_ease'] = df['review_details'].apply(textstat.flesch_reading_ease)
    df['smog_index'] = df['review_details'].apply(textstat.smog_index)
    df['flesch_kincaid_grade'] = df['review_details'].apply(textstat.flesch_kincaid_grade)
    df['coleman_liau_index'] = df['review_details'].apply(textstat.coleman_liau_index)
    df['automated_readability_index'] = df['review_details'].apply(textstat.automated_readability_index)
    df['dale_chall_readability_score'] = df['review_details'].apply(textstat.dale_chall_readability_score)
    df['difficult_words'] = df['review_details'].apply(textstat.difficult_words)
    df['linsear_write_formula'] = df['review_details'].apply(textstat.linsear_write_formula)
    df['gunning_fog'] = df['review_details'].apply(textstat.gunning_fog)
    df['text_standard'] = df['review_details'].apply(textstat.text_standard)

    # Sentiment
    df['text_sentiment'] = df['review_details'].apply(lambda x: get_review_sentiment(x))

    return df

def clean_username(name):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', name)
    cleaned = re.sub(r'[.|,|)|(|\|/|@]', r' ', cleaned)
    cleaned = re.sub(r'.csv', r'', cleaned)
    cleaned = re.sub(r'\d+', r'', cleaned)

    return cleaned

def name_extract_features(name):
    name = name.lower()
    return {
        'last_char': name[-1],
        'last_two': name[-2:],
        'last_three': name[-3:],
        'first': name[0],
        'first2': name[:1]
    }

def gender_name_training(name):
    # preparing a list of examples and corresponding class labels.
    labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])

    random.shuffle(labeled_names)

    # we use the feature extractor to process the names data.
    featuresets = [(name_extract_features(n), gender)
               for (n, gender) in labeled_names]

    # Divide the resulting list of feature
    # sets into a training set and a test set.
    train_set, test_set = featuresets[500:], featuresets[:500]

    # The training set is used to train a new "naive Bayes" classifier.
    gender_clf = NaiveBayesClassifier.train(train_set)

    # save the model to disk
    filename = 'gender_name_model.sav'
    pickle.dump(gender_clf, open(filename, 'wb'))

    return gender_clf


def gender_features(df):

    # gender estimation from username
    df['clean_username'] = df['username'].apply(lambda x: clean_username(x)) #clean username
    gender_clf = pickle.load(open(os.pathconf(MODELS_PATH, 'gender_name_model.sav').sav, 'rb')) # load model
    df['gender_estimation'] = df['clean_username'].apply(lambda x: gender_clf.classify(name_extract_features(x)))

    # Sentiment
    df['text_sentiment'] = df['review_details'].apply(lambda x: get_review_sentiment(x))

    # syntax
    df['pos_tag'] = df['text_p'].apply(lambda x: pos_tag(x.split(" ")))
    df['tags'] = df['pos_tag'].apply(lambda x: [pos for word, pos in (x)])
    # # summing to larger POS groups
    df['Adj'] = df['tags'].apply(lambda x: x.count('JJ') + x.count('JJR') + x.count('JJS'))
    df['Verb'] = df['tags'].apply(
        lambda x: x.count('VB') + x.count('VBD') + x.count('VBG') + x.count('VBN') + x.count('VBP') + x.count('VBZ'))
    df['Noun'] = df['tags'].apply(lambda x: x.count('NN') + x.count('NNS') + x.count('NNP') + x.count('NNPS'))
    df['Adv'] = df['tags'].apply(lambda x: x.count('RB') + x.count('RBR') + x.count('RBS'))

    #politeness

    # found this https://github.com/ryandavila/new-politeness but doesn't run


    return df




def age_prediction(df, classifier='lr'):
    df.dropna(subset=['age'], inplace=True)

    # Data to use
    X = df['text_p'] #age features extraction
    y = df['age']

    # Results without oversampling and only cv - F1 macro
    # NB: 0.64, LR: 0.66, RF: 0.54

    # using synthetic oversampling technique
    smote = SMOTE('minority')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=33)

    X_sm, y_sm = smote.fit(X_train, y_train)

    if classifier == 'nb':
        nb(X_sm, X_test, y_sm, y_test)
    elif classifier == 'lr':
        lr(X_sm, X_test, y_sm, y_test)
    elif classifier == 'rf':
        rf(X_sm, X_test, y_sm, y_test)
    else:
        sgd(X_sm, X_test, y_sm, y_test)
    # param_grid = [{'logisticregression__C': [1, 10, 100, 1000]}
    # gs = GridSearchCV(pipe, param_grid)
    # gs.fit(X, y)


if __name__ == '__main__':
    df_comments = pd.read_csv(os.path.join(INPUT_PATH, INPUT_FILENAME))
    df_demographics = read_comments_from_files(DEMOGRAPHICS_PATH, user_profiles=True)
    # Join dataframes
    df = df_comments.merge(df_demographics, on='username', how='inner')

    # Gender Prediction Process
    gender_prediction(df, classifier='rf')

    # Age Prediction Process
    age_prediction(df, classifier='rf')

    print()
