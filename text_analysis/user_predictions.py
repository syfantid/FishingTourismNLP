import os

import pandas as pd
user_profile_analysis = True
INPUT_FILENAME = 'processed_dataframe.csv'

def setup_variables():
    if user_profile_analysis:
        input_filepath = 'output\\output_user_profiles'
    else:
        input_filepath = 'output\\output_business_profiles'
    return input_filepath

if __name__ == '__main__':
    input_filepath = setup_variables()
    df = pd.read_csv(os.path.join(input_filepath, INPUT_FILENAME))
    print()