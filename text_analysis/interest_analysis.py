import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams, pyplot

rcParams.update({'figure.autolayout': True})


from text_analysis.read_comments import read_comments_from_files


INPUT_PATH = 'output\\output_user_profiles'
INPUT_FILENAME = 'processed_dataframe.csv'
BADGES_PATH = 'data_collection\\output_badges'

# For badge details see here: https://www.tripadvisor.com/TripCollectiveBadges
reviewer_badges = ['New Reviewer', 'Reviewer', 'Senior Reviewer', 'Contributor', 'Senior Contributor', 'Top Contributor']
passport_badge = ['Passport']
helpful_badge = ['Helpful Reviewer']
readership_badge = ['Readership']
explorer_badge = ['Explorer']
# All remaining badges are expertise badges

def adjust_graded_levels(value_counts, levels):
    original_value_counts = value_counts.copy()
    for level_1, level_2 in zip(levels,levels[1:]):
        value_counts[level_1] -= value_counts[level_2]
    return value_counts.sort_values()



def get_expertise_badges(df_badges):
    all_badges = [reviewer_badges, passport_badge, helpful_badge, readership_badge, explorer_badge]
    all_badges = sum(all_badges, [])
    expertise_badges = df_badges[~df_badges['badge'].isin(all_badges)]
    value_counts = expertise_badges.badge.value_counts()
    photographer_levels = ['New Photographer', 'Beginner Photographer', 'Junior Photographer', 'Photographer', 'Senior Photographer', 'Top Photographer']
    value_counts = adjust_graded_levels(value_counts, photographer_levels)
    total_counts = len(set(df_badges.username))
    value_percentages = value_counts / total_counts
    plt.barh(value_counts.index, value_percentages)
    plt.xlabel("Percentage of users holding the badge")
    plt.title("% of users per Expertise Badge")
    plt.savefig(os.path.join(INPUT_PATH, 'user_expertise_badges.jpg'))
    plt.clf()


def get_explorer_badges(df_badges):
    user_count = len(set(df_badges.username))
    explorer_badges = df_badges[df_badges['badge'].isin(explorer_badge)]
    print("{}\% of users having reviewed a fishing tourism business have an Explorer Badge".format(len(explorer_badges)*100/user_count))


def get_reviewer_badges(df_badges):
    all_reviewer_badges = df_badges[df_badges['badge'].isin(reviewer_badges)]
    value_counts = all_reviewer_badges.badge.value_counts()
    value_counts = adjust_graded_levels(value_counts, reviewer_badges)
    total_counts = sum(value_counts)
    value_percentages = value_counts / total_counts

    plt.barh(np.array(value_counts.index), np.array(value_percentages))
    plt.xlabel("Percentage of users holding the badge")
    plt.title("% of users per Reviewer Badge")
    plt.savefig(os.path.join(INPUT_PATH, 'reviewer_badges.jpg'))
    plt.clf()


def convert_to_numeric_index(s):
    l = [int(item) for subitem in s.index for item in subitem.split() if item.isdigit()]
    s.index = l
    return s


def get_passport_badges(df_badges):
    all_passport_badges = df_badges[df_badges['badge'].isin(passport_badge)]
    value_counts = all_passport_badges.subtext.value_counts()
    value_counts = convert_to_numeric_index(value_counts)
    # value_counts.index = pd.to_numeric(value_counts.index, errors='coerce')
    value_counts.sort_index(inplace=True)

    # total_counts = sum(value_counts)
    # value_percentages = value_counts / total_counts


    plt.plot(value_counts)
    plt.xlabel("Number of destinations visited")
    plt.ylabel("Number of users")
    plt.title("Number of destinations visited per user (Passport Badge)")
    # plt.show()
    # counts = int(value_counts.index)
    plt.savefig(os.path.join(INPUT_PATH, 'passport_badges.jpg'))
    plt.clf()

if __name__ == '__main__':
    # df_comments = pd.read_csv(os.path.join(INPUT_PATH, INPUT_FILENAME))
    df_badges = read_comments_from_files(BADGES_PATH, user_profiles=True)
    df_badges = df_badges.loc[:, ~df_badges.columns.str.contains('^Unnamed')]
    df_badges['badge_subtext'] = df_badges['badge'] + ' ' + df_badges['subtext']

    # Get information about passport badges
    get_passport_badges(df_badges)

    # Get information about reviewer badges
    get_reviewer_badges(df_badges)
    #
    # # Get information about expertise badges
    get_expertise_badges(df_badges)
    #
    # Get information about explorer badges
    get_explorer_badges(df_badges)
