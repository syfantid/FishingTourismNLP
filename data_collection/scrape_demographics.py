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

options = Options()
# options.add_argument('--headless')
options.add_argument('--hide-scrollbars')
options.add_argument('--disable-gpu')
options.add_argument('lang=en')


# driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
# driver = webdriver.Firefox(GeckoDriverManager().install(), options=options)

driver = Firefox(executable_path="C:\\Users\\Sofia\\Downloads\\geckodriver-v0.28.0-win64\\geckodriver.exe",
                 firefox_options=options)

# import the webdriver, chrome driver is recommended
driver.set_page_load_timeout(2)
filename = ""
i = 0


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
    regex = r"(man|woman)"
    return regex_match(regex, popup_text)


def extract_location(popup_text):
    regex = r"from (.*)"
    return regex_match(regex, popup_text)


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
        print("\nNo cookie pop-up!")

def get_user_popup(url):
    """
    Get the demographics (if available from a user's profile
    :param url: The user's profile URL to be scraped
    :return: None, simply calls function to write demographics to file
    """
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
        return

    # open up review in new tab
    review_summary = review.find_element_by_xpath(".//div[contains(@class, '_1kKLd-3D')]/a").get_attribute(
        "href")
    driver.execute_script("window.open('');")
    driver.switch_to.window(driver.window_handles[1])
    driver.get(review_summary)
    time.sleep(2)

    # scroll the viewpoint to the review and open up the pop-up window
    popup_element_link = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "expand_inline.scrname")))
    driver.execute_script("arguments[0].scrollIntoView();", popup_element_link)
    ActionChains(driver).move_to_element(popup_element_link).click().perform()
    time.sleep(1)

    # move the mouse to the pop-up to stay open
    popup_element = driver.find_elements_by_xpath("//div[@class='memberOverlayRedesign g10n']")[0]
    ActionChains(driver).move_to_element(popup_element).perform()
    time.sleep(1)

    # Get the pop-up content
    if check_exists_by_xpath("//div[@class='memberOverlayRedesign g10n']"):
        try:
            popup_content = driver.find_elements_by_class_name("innerContent")[0]
            popup_text = popup_element.text
            tags = get_tags()
            gender, age, location, member_since, cities_visited, contributions = extract_popup_info(popup_text)
            return gender, age, location, member_since, cities_visited, contributions, tags
        except Exception as e:
            print("Problem with extracting data from pop-up window for user " + url + ": " + str(e))
    else:
        print("Problem with opening up the pop-up window for user " + url)

    driver.close()
    driver.switch_to.window(driver.window_handles[0])
    return None


def get_user_profile_by_url(url):
    """
    Based on the given URL, we scrape the respective user's profile
    :param url: The profile URL to be scraped
    :return: None, simply handles the scraping process and writes to file
    """
    print("\nScraping: " + url)
    # get the name of place for csv file name
    global filename

    driver.get(url)
    driver.maximize_window()

    # username as the filename
    username = driver.find_element_by_class_name("gf69u3Nd").text
    filename = os.path.join('output_demographics', username + ".csv")

    return get_user_popup(url)


def get_all_user_demographics():
    # Reads all user profiles' URLs from the users who have reviewed fishing tourism businesses
    URLs = read_comments_from_files()['reviewer_profile']
    # URLs = ['https://www.tripadvisor.com/Profile/130Doug'] # test URL
    # Reads a URL at a time and calls the scraping function
    df = pd.DataFrame(columns=["url", "gender", "age", "location", "member_since", "cities_visited", "contributions"])
    for url in tqdm(URLs): # correct order
        try:
            driver.set_page_load_timeout(10)
            gender, age, location, member_since, cities_visited, contributions, tags = get_user_profile_by_url(url)
            print()
            print("url",url, "gender:",gender, "age:",age, "location:",location, "member_since:",member_since,
                            "cities_visited:",cities_visited, "contributions:",contributions, "tags:",tags)
            df = df.append({"url":url, "gender":gender, "age":age, "location":location, "member_since":member_since,
                            "cities_visited":cities_visited, "contributions":contributions, "tags":tags},
                           ignore_index=True)
        except TimeoutException as e:
            isrunning = 0
            print('\nThere is an issue, check again ' + url + " & Exception: " + str(e))
            # driver.close()

        print()

    write_to_file('demographics.csv', df, output_path='output_demographics')
    print('\nProgram is complete.')
    driver.close()


