import errno
import os
import re

import pandas as pd
from pathlib import Path


# For Mac
# INPUT_PATH = 'FishingTourismNLP/data_collection/output_reviews_new'

def get_username_from_filename(path_str):
    regex = r"[\w-]+.csv$"
    username = re.search(regex, path_str).group()
    return username


def read_comments_from_files(input, user_profiles=False):
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))  # This is your Project Root

    absolute_input_path = os.path.join(ROOT_DIR, input)

    df = pd.DataFrame()

    path_list = Path(absolute_input_path).rglob('*.csv')
    for path in path_list:
        path_str = str(path)  # because path is object not string
        file_df = pd.read_csv(path_str)
        if user_profiles:
            username = get_username_from_filename(path_str)
            file_df['username'] = username
        if len(df) != 0:
            df = pd.concat([df, file_df])
        else:
            df = file_df

    return df.reset_index(drop=True)
