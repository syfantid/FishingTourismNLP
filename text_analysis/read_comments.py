import errno
import os

import pandas as pd
from pathlib import Path

INPUT_PATH = 'FishingTourismNLP/data_collection/output_reviews'

def read_comments_from_files():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))  # This is your Project Root

    absolute_input_path = os.path.join(ROOT_DIR, INPUT_PATH)

    df = pd.DataFrame()

    path_list = Path(absolute_input_path).rglob('*.csv')
    for path in path_list:
        path_str = str(path) # because path is object not string
        file_df = pd.read_csv(path_str, index_col=0)
        if len(df) != 0:
            df = pd.concat([df, file_df])
        else:
            df = file_df

    return df.reset_index(drop=True)
