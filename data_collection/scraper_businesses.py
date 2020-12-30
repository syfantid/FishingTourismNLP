"""
This is a Python script to scrape business profiles from TripAdvisor based on a list of URLs.
"""

import math
import os
import time
import re
import urllib.parse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
import requests

# File paths
# INPUT_PATH = os.path.join(os.getcwd(), os.path.join("data_collection", "input"))
INPUT_PATH = os.path.join(os.getcwd(), "input")
OUTPUT_PATH = os.path.join(os.getcwd(), "output_reviews_new")
# Scraping classes
PAGE_CLASS = "ui_pagination"
NEXT_BUTTON_CLASS = "ui_button nav next primary"
REVIEW_CLASS = "Dq9MAugU T870kzTX LnVzGwUB"
REVIEW_TITLE_CLASS = "glasR4aX"
REVIEW_TEXT_CLASS = "cPQsENeY"
REVIEW_EXPANDED_TEXT_CLASS = "prw_rup prw_reviews_resp_sur_review_text_expanded"
REVIEW_DATE_CLASS = "_34Xs-BQm"
REVIEW_RATING_CLASS = "ui_bubble_rating"
REVIEW_RATING_1_CLASS = "ui_bubble_rating bubble_10"
REVIEW_RATING_2_CLASS = "ui_bubble_rating bubble_20"
REVIEW_RATING_3_CLASS = "ui_bubble_rating bubble_30"
REVIEW_RATING_4_CLASS = "ui_bubble_rating bubble_40"
REVIEW_RATING_5_CLASS = "ui_bubble_rating bubble_50"
REVIEW_LINK_CLASS = "ocfR3SKN"


def read_page_urls(input_filename):
    """
    Function to read enterprises' names and page URLs from the input file
    :param input_filename: The input file containing the enterprises' URLs
    :return: A list of enterprise names and a list of enterprise URLs
    """
    input_filepath = os.path.join(INPUT_PATH, input_filename)
    fishing_facilities = pd.read_excel(input_filepath)
    return fishing_facilities['enterprise_name'], fishing_facilities['ta_link']


def check_pagination_exists(soup):
    """
    Checks if the enterprise has multiple pages of reviews
    :param soup: The enterprise's page in HTML-Beautiful Soup format
    :return: The next page's URL I/A, None otherwise
    """
    regex = r"" + NEXT_BUTTON_CLASS + "\""
    if re.search(regex, str(soup)):
        next_div = soup.find_all("a", class_=NEXT_BUTTON_CLASS)[0]
        next_url = urllib.parse.urljoin("https://www.tripadvisor.com", next_div.get("href"))
        print("Next page URL: " + next_url)
        return next_url
    else:
        return None


def get_rating(full_review):
    """
    Function to get the rating from a single review
    :param full_review: The review's div in HTML-Beautiful Soup format
    :return: The numerical rating between 1 and 5
    """
    regex = r"ui_bubble_rating bubble_(\d)0"
    rating = re.search(regex, str(full_review)).group(1)
    return rating


def get_date(full_review):
    """
    Function to get the date of experience from a single review
    :param full_review: The review's div in HTML-Beautiful Soup format
    :return: The date of experience as a string in the format "Month Year", empty string if date of experience N/A
    """
    regex = r"((January|February|March|April|May|June|July|August|September|October|November|December) \d{4})"
    try:
        date = re.search(regex, str(full_review)).group(1)
    except:
        return ""
    return date


def get_reviewer_profile(full_review):
    """
    Function to get the reviewer's profile URL from a single review
    :param full_review: The review's div in HTML-Beautiful Soup format
    :return: The URL of the reviewer's profile
    """
    for link in full_review.find_all('a'):
        regex = r"Profile"
        if re.search(regex, str(full_review)):
            profile_url = urllib.parse.urljoin("https://www.tripadvisor.com", link.get('href'))
            return profile_url


def get_review_title(full_review):
    """
    Function to get the title from a single review
    :param full_review: The review's div in HTML-Beautiful Soup format
    :return: The review's title
    """
    title_div = full_review.find_all("div", class_=REVIEW_TITLE_CLASS)[0]
    title = title_div.get_text()
    return title


def get_review_text(full_review):
    """
    Function to get the text from a single review
    :param full_review: The review's div in HTML-Beautiful Soup format
    :return: \the review's text
    """
    review_link = full_review.find_all("a", class_=REVIEW_LINK_CLASS)[0].attrs['href']
    review_link = urllib.parse.urljoin("https://www.tripadvisor.com", review_link)
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(review_link, headers=headers)

    if response.status_code == 200:
        html = response.text
        # Convert to Beautiful Soup object
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.find("div", class_=REVIEW_EXPANDED_TEXT_CLASS).text
    else:
        text = ""
        # Convert the response HTLM string into a python string

    return text


def write_to_file(output_filename, reviews, output_path=OUTPUT_PATH):
    """
    Function to append reviews to file
    :param output_filename: The name of the output file to write the reviews
    :param reviews: The reviews to be appended
    :return: None
    """
    output_filepath = os.path.join(output_path, output_filename)
    if Path(output_filepath).is_file():
        reviews.to_csv(output_filepath, mode='a', header=False)
    else:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        reviews.to_csv(output_filepath)


def extract_reviews(output_filename, soup):
    """
    Function to extract all reviews from a single page
    :param output_filename: The name of the output file to write the reviews 
    :param soup: The enterprise's page in HTML-Beautiful Soup format
    :return: None
    """
    # extract reviewer, rating, date, review title and review text
    column_names = ["reviewer_profile", "date", "rating", "title", "text"]
    reviews = pd.DataFrame(columns=column_names)
    # Iterate over page's reviews
    full_reviews = soup.find_all("div", class_=REVIEW_CLASS)
    for full_review in full_reviews:
        rating = get_rating(full_review)
        date = get_date(full_review)
        reviewer_profile = get_reviewer_profile(full_review)
        title = get_review_title(full_review)
        text = get_review_text(full_review)
        review_series = pd.Series([reviewer_profile, date, rating, title, text], index=column_names)
        reviews = reviews.append(review_series, ignore_index=True)
        print(title)
        print(text)
        print("---------------------------------------------------")

    # Write to file
    write_to_file(output_filename, reviews)


def scrape_enterprise(output_filename, url):
    """
    Function to scrape an enterprise's TripAdvisor profile for reviews
    :param output_filename: The name of the output file to write the reviews 
    :param url: The URL of the enterprise
    :return: None
    """
    # Connection to web page
    headers = {'User-Agent': 'Mozilla/5.0'}
    while True:
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                html = response.text
                # Convert to Beautiful Soup object
                soup = BeautifulSoup(html, 'html.parser')
                # print(soup.prettify())
                extract_reviews(output_filename, soup)
                break
            else:
                print("Response Code: " + str(response.status_code))
                time.sleep(10)
                # Convert the response HTLM string into a python string

        next_url = check_pagination_exists(soup)
        if next_url is None:
            break
        else:
            url = next_url


def scrape_enterprises(input_filename):
    """
    Function to scrape all given enterprises' TripAdvisor profiles
    :param input_filename: The input filename containing the enterprises' details
    :return: None
    """
    enterprise_names, urls = read_page_urls(input_filename)
    for enterprise_name, url in zip(enterprise_names, urls):
        if not pd.isnull(url):
            scrape_enterprise(enterprise_name.strip(" ") + ".csv", url)
            time.sleep(1)

scrape_enterprises("Fishing_Vessels_FT_f.xls")