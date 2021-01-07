import os
from pprint import pprint
from time import time

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

INPUT_PATH = 'output\\output_user_profiles'
INPUT_FILENAME = 'processed_dataframe.csv'
DEMOGRAPHICS_PATH = 'data_collection\\output_demographics'


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


def gender_prediction_nb(X_train, X_test, y_train, y_test):
    pipeline_nb = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', MultinomialNB())
    ])

    clf_nb = pipeline_nb.fit(X_train, y_train)
    y_predicted = clf_nb.predict(X_test)
    evaluate(y_test, y_predicted, clf_nb, X_test, "Multinomial Bernoulli")


def gender_prediction_lr(X_train, X_test, y_train, y_test):
    pipeline_lr = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression())
    ])

    clf_lr = pipeline_lr.fit(X_train, y_train)
    y_predicted = clf_lr.predict(X_test)
    evaluate(y_test, y_predicted, clf_lr, X_test, "Logistic Regression")


def gender_prediction_rf(X_train, X_test, y_train, y_test):
    pipeline_rf = Pipeline([
        ('vect', CountVectorizer()),
        ('lr', RandomForestClassifier())
    ])

    clf_rf = pipeline_rf.fit(X_train, y_train)
    y_predicted = clf_rf.predict(X_test)
    evaluate(y_test, y_predicted, clf_rf, X_test, "Random Forest", X_train=X_train, y_train=y_train)


def gender_prediction_sgd(X_train, X_test, y_train, y_test):
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
        gender_prediction_nb(X_train, X_test, y_train, y_test)
    elif classifier == 'lr':
        gender_prediction_lr(X_train, X_test, y_train, y_test)
    elif classifier == 'rf':
        gender_prediction_rf(X_train, X_test, y_train, y_test)
    else:
        gender_prediction_sgd(X_train, X_test, y_train, y_test)
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

    print()
