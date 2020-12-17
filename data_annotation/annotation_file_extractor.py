import errno
import os

import pandas as pd
from pathlib import Path

INPUT_PATH = 'data_collection\\output_reviews'

def read_comments_from_files():
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
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


def get_df_subset(df, size):
    return df.sample(n=size, replace=False)


def create_annotation_files(df_sample, examples_number, annotators_number):
    output_dir_name = "output_to_annonate"
    try:
        os.mkdir(output_dir_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    labels_names = ["Education", "Entertainment", "Aestheticism", "Escape"]

    for file_counter in range(annotators_number):
        filepath = os.path.join(os.path.join(os.getcwd(), output_dir_name), "file_" + str(file_counter) + ".xlsx")
        df_file_sample = df_sample.sample(n=examples_number, replace=False)[["text", "title"]]
        df_file_sample = pd.concat([df_file_sample, pd.DataFrame(columns=labels_names)], axis=1)
        df_file_sample.to_excel(filepath, index=False, index_label=False)


def get_annotation_files():
    df = read_comments_from_files()
    df_sample = get_df_subset(df, 200)
    create_annotation_files(df_sample, 60, 5)

get_annotation_files()