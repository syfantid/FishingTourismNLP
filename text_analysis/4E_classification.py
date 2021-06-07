import os
import re

import pandas as pd
import seaborn as sns;

from utils import ROOT_DIR

sns.set_theme()
import matplotlib.pyplot as plt
import numpy as np

from text_analysis.read_comments import read_comments_from_files
from text_analysis.text_processing import processing_steps, n_grams, tf_idf, word_cloud, ngrams_cloud, \
    word_frequencies_graph
from itertools import combinations

INPUT_PATH = 'output\\output_4E'
INPUT_FILENAME = 'processed_dataframe.csv'
ANNOTATED_PATH = 'data_annotation\\annotated_files'
E_s = ["Education", "Entertainment", "Aestheticism", "Escape"]


def visualization(df, output_folder):
    # Vizualization and stats - word clouds, word bars etc
    output_filepath = os.path.join('output', 'output_4E', output_folder)
    if not os.path.exists(output_filepath):
        os.mkdir(output_filepath)
    word_cloud(df['text_p'], output_filepath)
    ngrams_cloud(df['text_p'], output_filepath)
    word_frequencies_graph(df['text_p'], output_filepath)


def remove_common_words(df):
    common_words = ['owner_name', 'tour_location', 'fishing', 'trip', 'fish', 'day']

    # Compile a regular expression that will match all the words in one sweep
    common_words_re = re.compile("|".join(r"\b%s\b" % re.escape(word) for word in common_words))

    # Replace and reassign into the column
    df["text_p"].replace(common_words_re, "", inplace=True)
    return df


def create_coexistence_heatmap(df):
    df_vis = pd.DataFrame(columns=E_s, index=E_s)
    e_pairs = list(combinations(E_s, 2))

    for (e1, e2) in e_pairs:
        df_subset = df.loc[(df[e1] == 1.0) & (df[e2] == 1.0)]
        df_vis.loc[e1, e2] = len(df_subset) / (
                len(df.loc[(df[e1] == 1.0)]) + len(df.loc[(df[e2] == 1.0)]) - len(df_subset))
        df_vis.loc[e2, e1] = df_vis.loc[e1, e2]

    for e in E_s:
        df_vis.loc[e, e] = 1
    # df_vis = df_vis.astype(int)
    df_vis.fillna(value=np.nan, inplace=True)
    mask = np.triu(df_vis, k=1)
    sns.heatmap(df_vis, annot=True, fmt='g', mask=mask)
    plt.title("Heatmap of the Jaccard index of the 4E dimensions in the reviews' subset (N=240)")

    # corr = df_vis.corr()
    # sns.heatmap(corr, annot=True)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join("output", "output_4E", "jaccard_heatmap.png"))
    # plt.show()
    plt.clf()


def create_frequency_heatmap(df):
    df_vis = pd.DataFrame(columns=E_s, index=E_s)
    e_pairs = list(combinations(E_s, 2))

    for (e1, e2) in e_pairs:
        df_subset = df.loc[(df[e1] == 1.0) & (df[e2] == 1.0)]
        df_vis.loc[e1, e2] = len(df_subset) / len(df)
        df_vis.loc[e2, e1] = df_vis.loc[e1, e2]

    for e in E_s:
        df_vis.loc[e, e] = len(df.loc[(df[e] == 1.0)]) / len(df)
    # df_vis = df_vis.astype(int)
    df_vis.fillna(value=np.nan, inplace=True)
    mask = np.triu(df_vis, k=1)
    sns.heatmap(df_vis, annot=True, fmt='g', mask=mask)
    plt.title("Heatmap of the frequency of the 4E dimensions in the reviews' subset (N=240)")

    # corr = df_vis.corr()
    # sns.heatmap(corr, annot=True)
    plt.yticks(rotation=0)
    plt.savefig(os.path.join("output", "output_4E", "frequency_heatmap.png"))
    # plt.show()
    plt.clf()


def demographics_4e_visualization(df):
    pass


def marital_status_agg(s):
    counts = s.value_counts()
    if 'family' in counts and 'couple' in counts:
        return 'family' if counts.family >= counts.couple else 'couple'
    elif 'family' in counts:
        return 'family'
    elif 'couple' in counts:
        return 'couple'
    else:
        return 'unknown'


def aggregate_demographics_per_user(df):
    df_users = df.groupby(['username'])[['gender', 'age']].agg(pd.Series.mode)
    df_users['marital_status'] = df.groupby(['username'])[['marital_status']].agg(marital_status_agg)
    df_users.to_csv(os.path.join(ROOT_DIR, 'users_demographics.csv'))
    return df_users


# todo: Calculate interrater agreement
if __name__ == '__main__':
    df_demographics = pd.read_csv(os.path.join(ROOT_DIR, 'df_with_demographics.csv'))
    df_users = aggregate_demographics_per_user(df_demographics)
    print()

    # Visualize dimensions (4Es) coexistence
    # df = read_comments_from_files(ANNOTATED_PATH, user_profiles=True)
    # filepath = os.path.join(INPUT_PATH, 'processed_dataframe.csv')
    # if os.path.exists(filepath):
    #     df = pd.read_csv(filepath)
    # else:
    #     df = processing_steps(df, 'text', 'title')
    #     df.to_csv(filepath)
    #
    # create_coexistence_heatmap(df)
    # create_frequency_heatmap(df)

    # Visualization of word clouds per E dimension
    # df = remove_common_words(df)
    # for e in E_s:
    #     df_e = df[df[e] == 1]
    #     visualization(df_e, e)
