import os
import re
import string

import pandas as pd
import seaborn as sns

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


def demographics_4e_visualization(df_users):
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
def get_username(df):
    def get_username_from_url(url):
        regex = r"(?:https://www.tripadvisor.com/Profile/)(.*)"
        username = re.search(regex, url).group(1)
        return username + '.csv'

    df_all = pd.read_csv(os.path.join(ROOT_DIR, 'text_analysis', 'output', 'output_business_profiles', 'processed_dataframe.csv'))
    df_all['username'] = df_all.apply(lambda row: get_username_from_url(row['reviewer_profile']), axis=1)
    df_all = df_all[['username', 'text']]
    df_merged = df.merge(df_all, left_on='text', right_on='text', how='left')
    return df_merged


def visualize_demographics(df, e):
    # Visualize gender
    total = df['gender'].value_counts().sum()
    man_perc = df['gender'].value_counts().man/total
    woman_perc = df['gender'].value_counts().woman/total
    y = np.array([man_perc, woman_perc, max(0,1-man_perc-woman_perc)])
    mylabels = ["Man", "Woman", "N/A"]
    plt.pie(y, startangle=90, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title('Gender distribution for the {} dimension'.format(e))
    plt.legend(mylabels, bbox_to_anchor=(1,0), loc="lower right",
                          bbox_transform=plt.gcf().transFigure)
    # plt.legend(mylabels, loc="lower right")
    plt.savefig(os.path.join('output', 'output_4E', 'gender_{}.png'.format(e)))
    plt.clf()

    # Visualize age group
    total = df['age'].value_counts().sum()
    a_35_49 = df['age'].value_counts()['35-49'] / total
    a_50_64 = df['age'].value_counts()['50-64'] / total
    a_25_34 = df['age'].value_counts()['25-34'] / total
    y = np.array([a_25_34, a_35_49, a_50_64])
    mylabels = ["25-34", "35-49", "50-64"]
    plt.pie(y, startangle=90, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title('Age distribution for the {} dimension'.format(e))
    plt.legend(mylabels, loc="lower right")
    plt.savefig(os.path.join('output', 'output_4E', 'age_{}.png'.format(e)))
    plt.clf()

    # Visualize marital status
    total = df['marital_status'].value_counts().sum()
    a_couple = df['marital_status'].value_counts()['couple'] / total
    a_family = df['marital_status'].value_counts()['family'] / total
    a_unknown = df['marital_status'].value_counts()['unknown'] / total
    y = np.array([a_couple, a_family, a_unknown])
    mylabels = ["Couple", "Family", "N/A"]
    plt.pie(y, startangle=90, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.2)
    plt.title('Marital status distribution for the {} dimension'.format(e))
    plt.legend(mylabels, loc="lower right")
    plt.savefig(os.path.join('output', 'output_4E', 'marital_status_{}.png'.format(e)))
    plt.clf()

def random_selection_for_ambiguous_users(df_users):
    df_users['age'] = df_users.apply(lambda row: row['age'][0] if isinstance(row['age'], np.ndarray) else row['age'], axis=1)
    df_users['gender'] = df_users.apply(lambda row: row['gender'][0] if isinstance(row['gender'], np.ndarray) else row['gender'], axis=1)
    return df_users


if __name__ == '__main__':
    # Read reviews from annotated files
    filepath = os.path.join(INPUT_PATH, 'processed_dataframe.csv')
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
    else:
        df = read_comments_from_files(ANNOTATED_PATH, user_profiles=True)
        df = processing_steps(df, 'text', 'title')
        df.to_csv(filepath)
    # Get username for annotated reviews
    df = get_username(df)

    # Read user demographics
    df_demographics = pd.read_csv(os.path.join(ROOT_DIR, 'df_with_demographics.csv'))
    df_demographics['gender'] = df_demographics.apply(lambda row: row['gender'].lower(), axis=1)
    # Aggregate demographics per user
    df_users = aggregate_demographics_per_user(df_demographics)
    df_users = random_selection_for_ambiguous_users(df_users)

    # Aggregate demographics with the annotated reviews
    df_with_demographics = df.merge(df_users, left_on='username_y', right_index=True, how='left')

    for e in E_s:
        df_e = df_with_demographics[df_with_demographics[e] == 1]
        visualize_demographics(df_e, e)

    # Visualize dimensions (4Es) coexistence
    # create_coexistence_heatmap(df)
    # create_frequency_heatmap(df)

    # Visualization of word clouds per E dimension
    # df = remove_common_words(df)
    # for e in E_s:
    #     df_e = df[df[e] == 1]
    #     visualization(df_e, e)
