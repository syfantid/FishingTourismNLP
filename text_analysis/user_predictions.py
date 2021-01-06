import os

import pandas as pd

from text_analysis.read_comments import read_comments_from_files

INPUT_PATH = 'output\\output_user_profiles'
INPUT_FILENAME = 'processed_dataframe.csv'
DEMOGRAPHICS_PATH = 'data_collection\\output_demographics'


def gender_prediction(df):
    df.dropna(subset=['gender'], inplace=True)
    print()


if __name__ == '__main__':
    df_comments = pd.read_csv(os.path.join(INPUT_PATH, INPUT_FILENAME))
    df_demographics = read_comments_from_files(DEMOGRAPHICS_PATH, user_profiles=True)
    # Join dataframes
    df = df_comments.merge(df_demographics, on='username', how='inner')

    # Gender Prediction Process
    gender_prediction(df)

    print()