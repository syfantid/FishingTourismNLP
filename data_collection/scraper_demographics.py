"""
This is an adapted Python script for TripAdvisor user profile scraping from Sanjanaekanayake's profile:
https://github.com/Sanjanaekanayake/tripadvisor-user-profiles-scrapper/blob/main/getUserReviews.py

Changes from the original script:
* No limit for review count (previously 100)
* Extra data extracted per review (rating and location)
* Fixed bugs in the scraping process (removed try-catch block that concealed issues)
* Added support for browser options to overcome the timeout issue caused in the latest release of Chrome driver
* Added support for Firefox driver
* Added progress bards to monitor scraper's progress
"""
import errno
import os
import csv
import re
import time
from tqdm import tqdm
import pandas as pd

# import the webdriver, chrome driver is recommended
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from data_annotation.annotation_file_extractor import read_comments_from_files
from data_collection.scraper_businesses import write_to_file
from data_collection.scraper_user_badges import get_username_from_url

options = Options()
options.add_argument('--headless')
options.add_argument('--hide-scrollbars')
options.add_argument('--disable-gpu')
# options.add_argument('--lang=en')
options.set_preference('intl.accept_languages', 'en-GB')

# driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
# driver = webdriver.Firefox(GeckoDriverManager().install(), options=options)

# driver = Firefox(executable_path="C:\\Users\\Sofia\\Downloads\\geckodriver-v0.28.0-win64\\geckodriver.exe",
#                  firefox_options=options)

#to run on mac

from selenium import webdriver
driver = webdriver.Chrome(executable_path='/usr/local/bin/chromedriver')


# import the webdriver, chrome driver is recommended
driver.set_page_load_timeout(2)
filename = ""
i = 0

def correct_file_format():
    newline = os.linesep  # Defines the newline based on your OS.

    input_path = 'data_collection\\output_demographics'
    output_path = 'data_collection\\output_demographics_new'
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root
    absolute_input_path = os.path.join(ROOT_DIR, input_path)
    absolute_output_path = os.path.join(ROOT_DIR, output_path)
    make_directory(absolute_output_path)

    for filename in os.listdir(absolute_input_path):
        source_fp = open(os.path.join(absolute_input_path, filename), 'r', encoding='utf-8')
        new_filename = filename + '.csv'
        target_fp = open(os.path.join(absolute_output_path, new_filename), 'w', encoding='utf-8')
        first_row = True
        for row in source_fp:
            if first_row:
                row = 'gender,age,location,member_since,cities_visited,contributions,tags,popup_text'
                first_row = False
                target_fp.write(row + '\n')
            else:
                row = row.replace('\n',' ')
                row = row.replace(', ', '-')
                target_fp.write(row)
        source_fp.close()
        target_fp.close()

        # with open(os.path.join(absolute_input_path, filename), 'r') as f:  # open in readonly mode
        #     print()
    # do your stuff

def check_exists_by_xpath(xpath):
    """
    Function to check if button exists in HTML to avoid miss-clicks
    :param xpath: The button's XPATH expression to be evaluated
    :return: True, if the button exists, false otherwise
    """
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
    time.sleep(2)


def regex_match(regex, text):
    try:
        result = re.search(regex, text)
        try:
            result = result.group(1)
        except IndexError:
            result = result.group(0)
    except:
        result = ""
    return result


def extract_member_since(popup_text):
    regex = (r"Tripadvisor member since ([0-9]{4})\n")
    return regex_match(regex, popup_text)


def extract_age(popup_text):
    regex = r"([0-9]{2}-[0-9]{2})"
    return regex_match(regex, popup_text)


def extract_gender(popup_text):
    regex = r"(\b[m|M]an\b|\b[w|W]oman\b)"
    return regex_match(regex, popup_text)


def extract_location(popup_text):
    regex = r"[f|F]rom (.*)"
    return regex_match(regex, popup_text).replace(', ','-')


def extract_cities_visited(popup_text):
    regex = r"(\d+) Cities visited"
    return regex_match(regex, popup_text)


def extract_contributions(popup_text):
    regex = r"(\d+) Contributions"
    return regex_match(regex, popup_text)


def extract_popup_info(popup_text):
    gender = extract_gender(popup_text)
    age = extract_age(popup_text)
    location = extract_location(popup_text)
    member_since = extract_member_since(popup_text)
    cities_visited = extract_cities_visited(popup_text)
    contributions = extract_contributions(popup_text)
    return gender, age, location, member_since, cities_visited, contributions


def get_tags():
    tags_list = driver.find_elements_by_class_name("memberTagReviewEnhancements")
    tags_text = []
    for tag in tags_list:
        tags_text.append(tag.text)
    return tags_text


def check_cookie_popup():
    try:
        element_privacy = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "evidon-banner-acceptbutton")))
        ActionChains(driver).move_to_element(element_privacy).click().perform()
    except:
        # print("\nNo cookie pop-up!")
        print()


def get_user_profile_by_url(url):
    """
        Get the demographics (if available from a user's profile
        :param url: The user's profile URL to be scraped
        :return: None, simply calls function to write demographics to file
    """
    print("\nScraping: " + url)
    # try:
    #     driver.get(url)
    #     driver.maximize_window()
    #     check_cookie_popup()
    # except Exception:
    #     print("Cannot get demographics for this user")
    #     return []

    driver.get(url)
    driver.maximize_window()
    check_cookie_popup()

    # go to reviews' tab
    reveiwstab = driver.find_element_by_xpath('//a[@data-tab-name="Reviews"]')
    reveiwstab.click()
    time.sleep(2)

    # get the first review
    try:
        review = driver.find_elements_by_xpath("//div[@style='position:relative']/div")[0]
    except Exception:
        print("User has no reviews.")
        return []

    # open up review in new tab
    review_summary = review.find_element_by_xpath(".//div[contains(@class, '_1kKLd-3D')]/a").get_attribute(
        "href")
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get(review_summary)
    time.sleep(2)

    # check if date overlay exists
    if check_exists_by_xpath("//div[@class='rsdc-wrapper corgi rsdc-dual-month']"):
        try:
            popup_element_link = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "featured-review-container")))
            driver.execute_script("arguments[0].scrollIntoView();", popup_element_link)
            ActionChains(driver).move_to_element(popup_element_link).click().perform()
            time.sleep(1)
        except:
            print()


    # scroll the viewpoint to the review and open up the pop-up window
    if check_exists_by_xpath("//span[@class='expand_inline scrname']"):
        popup_element_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "expand_inline.scrname")))
        driver.execute_script("arguments[0].scrollIntoView();", popup_element_link)
        ActionChains(driver).move_to_element(popup_element_link).click().perform()
        time.sleep(1)
    elif check_exists_by_xpath("//div[@class='memberOverlayLink']"):
        popup_element_link = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "memberOverlayLink")))
        driver.execute_script("arguments[0].scrollIntoView();", popup_element_link)
        ActionChains(driver).move_to_element(popup_element_link).click().perform()
        time.sleep(1)
    elif check_exists_by_xpath("//div[@class='info_text']"):
        popup_element_link = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.CLASS_NAME, "info_text")))
        driver.execute_script("arguments[0].scrollIntoView();", popup_element_link)
        ActionChains(driver).move_to_element(popup_element_link).click().perform()
        time.sleep(1)

    # move the mouse to the pop-up to stay open
    try:
        popup_element = driver.find_elements_by_xpath("//div[@class='memberOverlayRedesign g10n']")[0]
        ActionChains(driver).move_to_element(popup_element).perform()
    except:
        try:
            popup_element = driver.find_elements_by_xpath("//div[@class='memberOverlayRedesign g10n']")
            ActionChains(driver).move_to_element(popup_element).perform()
        except:
            popup_element = driver.find_elements_by_xpath("//span[@class='ui_overlay ui_popover arrow_left ']")
            ActionChains(driver).move_to_element(popup_element).perform()
            print()
    time.sleep(1)

    # Get the pop-up content
    if check_exists_by_xpath("//div[@class='memberOverlayRedesign g10n']"):
        try:
            popup_content = driver.find_elements_by_class_name("innerContent")[0]
            popup_text = popup_element.text
            tags = get_tags()
            gender, age, location, member_since, cities_visited, contributions = extract_popup_info(popup_text)
            print("\nPop-up Text: " + popup_text)
            print("\nGender: " + gender + " - Age: " + age + " - Location: " + location
                  + " - Member Since: " + member_since + " - Cities: " + cities_visited + " - Contributions: "
                  + contributions)
            return [gender, age, location, member_since, cities_visited, contributions, tags, popup_text]
        except Exception as e:
            print("Problem with extracting data from pop-up window for user " + url + ": " + str(e))
    else:
        print("Problem with opening up the pop-up window for user " + url)

    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return []


def make_directory(directory_name):
    try:
        os.mkdir(directory_name)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_all_user_demographics():
    # Reads all user profiles' URLs from the users who have reviewed fishing tourism businesses
    URLs = read_comments_from_files()['reviewer_profile']

    columns = ["gender", "age", "location", "member_since", "cities_visited", "contributions", "tags", "popup_text"]

    # create output directory if it doesn't exist
    directory_name = "output_demographics"
    make_directory(directory_name)

    for url in tqdm(URLs[1018:]):  # correct order
        try:
            # Check if file already exists
            filename = os.path.join(directory_name, get_username_from_url(url))
            filename += ".csv"
            if os.path.isfile(filename):  # check if user's profile has been already parsed
                print("\nUser's profile is already parsed in " + filename)
                continue

            # Get user's data
            driver.set_page_load_timeout(15)
            demographics_list = get_user_profile_by_url(url)

            # Save user's data
            with open(filename, 'a', encoding='utf-8') as userfile:
                userfile.write(','.join(columns))  # write column names
                userfile.write('\n')
                userfile.write(','.join(str(v) for v in demographics_list))  # write demographics to file
        except TimeoutException as e:
            print('\nThere is an issue, check again ' + url + " & Exception: " + str(e))
            # driver.close()

    print('\nProgram is complete.')
    driver.close()

correct_file_format()