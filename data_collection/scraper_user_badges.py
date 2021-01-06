"""
This is a Python script to scrape users' badges profiles from TripAdvisor based on a list of user profile URLs.
"""
import os
import re
import urllib.parse
import time
import pandas as pd

import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

from data_annotation.annotation_file_extractor import read_comments_from_files
from data_collection.scraper_businesses import write_to_file

BASE_URL = 'https://www.tripadvisor.com/members-badgecollection/'
BADGE_TEXT_CLASS = 'badgeText'
BADGE_SUBTEXT_CLASS = 'subText'
BADGE_INFO_CLASS = 'badgeInfo'


def get_username_from_url(url):
    regex = r"[^/][\w-]+$"

    username = re.search(regex, url).group()
    return username


def get_badge_page_url(url):
    username = get_username_from_url(url)
    url = urllib.parse.urljoin(BASE_URL, username)
    return url


def extract_badges(filename, soup):
    badges = pd.DataFrame(columns=['badge', 'subtext'])
    all_badges_iterable = soup.find_all("div", class_=BADGE_INFO_CLASS)
    for badge_info in all_badges_iterable:
        try:
            badge_text_div = badge_info.find_all("div", class_=BADGE_TEXT_CLASS)[0]
            badge_subtext_div = badge_info.find_all("span", class_=BADGE_SUBTEXT_CLASS)[0]
            badge_text = badge_text_div.get_text()
            badge_subtext = badge_subtext_div.get_text()
            badges = badges.append({"badge": badge_text, "subtext": badge_subtext}, ignore_index=True)
        except IndexError:
            print("Cannot read badge " + str(badge_info) + " for file " + filename)
            continue

    write_to_file(filename, badges, output_path='output_badges')
    return badges


def get_user_badges(url):
    # Connection to web page
    headers = {'User-Agent': 'Mozilla/5.0'}
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            html = response.text
            # Convert to Beautiful Soup object
            soup = BeautifulSoup(html, 'html.parser')
            # print(soup.prettify())
            username = get_username_from_url(url)
            filename = username + '.csv'
            if os.path.isfile(os.path.join('output_badges', filename)):
                print("\nThe user's badge profile is already crawled.")
            else:
                extract_badges(filename, soup)
            break
        else:
            print("Response Code: " + str(response.status_code))
            time.sleep(10)
            # Convert the response HTLM string into a python string


def get_users_badges():
    # Reads all user profiles' URLs from the users who have reviewed fishing tourism businesses
    URLs = read_comments_from_files()['reviewer_profile']

    # Reads a URL at a time and calls the scraping function
    for url in tqdm(URLs):  # correct order
        print("\nGetting user's badge profile for " + url)
        badge_page_url = get_badge_page_url(url)
        get_user_badges(badge_page_url)
