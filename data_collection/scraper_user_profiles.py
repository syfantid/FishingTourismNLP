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

# import the webdriver, chrome driver is recommended
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from data_annotation.annotation_file_extractor import read_comments_from_files

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


def write_csv(review_title, review_text, review_date, review_for, review_rating, review_location):
    """
    Function to write review details to file
    :param review_title: The title of the review
    :param review_text: The text of the review
    :param review_date: The date the review was published
    :param review_for: The business the review refers to
    :param review_rating: The rating of the review
    :param review_location: The location of the business the review refers to
    :return: None, simply writes review to file
    """
    with open(filename, mode='a', newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([str(review_title), str(review_text), str(review_date), str(review_for), str(review_rating),
                         str(review_location)])


def write_log(username, error):
    """
    Writes error logs related to error in scraping user profiles
    :param username: The user profile the error refers to
    :param error: The error in the scraping process
    :return: None, simply writes the log file
    """
    filename = "scrape_bugs.log"
    with open(filename, mode='a', newline='') as l:
        writer = csv.writer(l)
        writer.writerow([str(username), str(error)])


def get_user_reviews(URL, end_count):
    """
    Get all reviews from a user's profile
    :param URL: The user's profile URL to be scraped
    :param end_count: The total number of reviews
    :return: None, simply calls function to write reviews to file
    """
    try:
        element_privacy = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CLASS_NAME, "evidon-banner-acceptbutton")))
        ActionChains(driver).move_to_element(element_privacy).click().perform()
    except:
        print("\nNo cookie pop-up!")

    reveiwstab = driver.find_element_by_xpath('//a[@data-tab-name="Reviews"]')
    reveiwstab.click()
    time.sleep(2)

    if check_exists_by_xpath("//div[@id='content']"):
        # to expand the review if show more button exists
        if check_exists_by_xpath("//span[@class='_1ogwMK0l']"):
            showmorebutton = driver.find_element_by_xpath("//span[@class='_1ogwMK0l']")
            showmorebutton.click()
            time.sleep(2)

    # Scrolls as much as possible to make all reviews appear and gets the total number of reviews
    while driver.find_elements_by_xpath("//div[@style='position:relative']/div"):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        review = driver.find_elements_by_xpath("//div[@style='position:relative']/div")
        element_count = len(review)
        # covers the cases where review count is different than the one mentioned in TripAdvisor's contributions pop-up
        if end_count - 19 < element_count < end_count + 19:
            print("\nReviews to be parsed: " + str(element_count))
            break
        else:
            continue
    if element_count is None:
        element_count = []
    # iteration over all reviews
    for j in tqdm(range(element_count)):
        # name = review[j].find_element_by_xpath(".//div[contains(@class, '_2fxQ4TOx')]").text
        # extract title
        try:
            review_title = review[j].find_element_by_xpath(".//div[contains(@class, '_3IEJ3tAK _2K4zZcBv')]").text
        except NoSuchElementException:
            print("\nCannot find title for this review.")
            review_title = ""
        # extract date (if available)
        try:
            review_date = review[j].find_element_by_xpath(".//div[contains(@class, '_3Coh9OJA')]").text
        except NoSuchElementException:
            print("\nCannot find date for this review.")
            review_date = ""
        # extract reviewed business
        review_for = review[j].find_element_by_xpath(".//div[contains(@class, '_2ys8zX0p ui_link')]").text
        # Used later to open up review in new tab
        try:
            review_summary = review[j].find_element_by_xpath(".//div[contains(@class, '_1kKLd-3D')]/a").get_attribute(
            "href")
        except NoSuchElementException:
            print("\nCannot find the element for this review.")
            continue
        # extract reviewed business' location
        try:
            review_location = review[j].find_element_by_xpath(".//div[contains(@class, '_7JBZK6_8 _20BneOSW')]").text
        except NoSuchElementException:
            print("\nCannot find location for this review.")
            review_location = ""
        # extract rating
        review_rating = 5
        if check_exists_by_xpath("//span[@class='ui_bubble_rating bubble_40']"):
            review_rating = 4
        elif check_exists_by_xpath("//span[@class='ui_bubble_rating bubble_30']"):
            review_rating = 3
        elif check_exists_by_xpath("//span[@class='ui_bubble_rating bubble_20']"):
            review_rating = 2
        if check_exists_by_xpath("//span[@class='ui_bubble_rating bubble_10']"):
            review_rating = 1

        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(review_summary)
        time.sleep(2)

        # Get review full text
        if check_exists_by_xpath("//span[@class='fullText hidden']"):
            try:
                read_more_button = driver.find_elements_by_xpath(
                    "//div[@class='reviewSelector']/div/div[2]/div[3]/div/p/span")
                read_more_button[2].click()
                review_text = read_more_button[1].text
            except IndexError:
                review_details = driver.find_elements_by_xpath("//span[@class='fullText hidden']")[0]
                review_text = review_details.text
                # print("\n--------------1st IF: " + review_text)
        elif check_exists_by_xpath("//span[@class='fullText ']"):
            try:
                read_more_button = driver.find_elements_by_xpath(
                    "//div[@class='reviewSelector']/div/div[2]/div[3]/div/p/span")
                review_text = read_more_button[0].text
            except IndexError:
                review_details = driver.find_elements_by_xpath("//span[@class='fullText ']")[0]
                review_text = review_details.text
            # print("\n************2nd IF: " + review_text)
        elif check_exists_by_xpath("//p[@class='partial_entry']"):
            review_details = driver.find_elements_by_xpath("//p[@class='partial_entry']")[0]
            review_text = review_details.text
            # print("\n-----------3rd IF: " + review_text)
        elif check_exists_by_xpath("//div[@class='entry vrReviewText']"):
            review_details = driver.find_elements_by_xpath("//div[@class='entry vrReviewText']")[0]
            review_text = review_details.text
            # print("\n^^^^^^^^^^^4rd IF: " + review_text)
        else:
            review_details = driver.find_elements_by_xpath(
                "//div[@class='reviewSelector']/div/div[2]/div/div/div[3]/div/p")
            try:
                review_text = review_details[0].text
            except IndexError:
                review_text = ""
                print("Cannot find text for this review.")
            # print("\n$$$$$$$$$$5th IF: " + review_text)

        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        # print("Review to be written to file: " + review_title)
        # print(review_text)
        write_csv(review_title, review_text, review_date, review_for, review_rating, review_location)


def get_user_profile_by_url(URL):
    """
    Based on the given URL, we scrape the respective user's profile
    :param URL: The profile URL to be scraped
    :return: None, simply handles the scraping process and writes to file
    """
    print("\nScraping: " + URL)
    # get the name of place for csv file name
    global filename

    driver.get(URL)
    driver.maximize_window()

    # get review count from the contributions pop-up element
    element = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CLASS_NAME, "_1q4H5LOk")))
    ActionChains(driver).move_to_element(element).click().perform()
    time.sleep(1)
    # The text "x reviews" where x is the number of user's reviews
    count_text_elements = driver.find_elements_by_class_name("_3wryStHh")
    # Flag to check if the user has any reviews
    reviews_available = False
    for element in count_text_elements:
        if 'reviews' in element.text or 'review' in element.text:
            count_text = element.text
            reviews_available = True
    if not reviews_available:
        print("\nThe user does not have any reviews.")
        return

    # The above string stripped of non-numeric characters and adjusted for thousands
    count = int(re.sub("[^0-9]", "", count_text).replace(",", ""))
    # Click X for the contributions pop-up
    driver.find_element_by_class_name("_2EFRp_bb._9Wi4Mpeb").click()
    time.sleep(1)

    # username as the filename
    username = driver.find_element_by_class_name("gf69u3Nd").text

    filename = os.path.join('output_profiles', username + ".csv")
    print('\nReady to scrape ' + username + "'s profile with " + str(count) + " reviews.")

    # open csv file and add titles only if they do not already exist
    if os.path.isfile(filename):
        print("\nUser profile is already parsed. Continuing!")
        return
    with open(filename, mode='w', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [str('reviewTitle'), str('review_details'), str('reviewDate'), str('reviewFor'), str("reviewRating"),
             str('reviewLocation')])

    end_count = count

    get_user_reviews(URL, end_count)
    print('\nSaved reviews in page = ', str(end_count), ' user = ', filename)
    print()


# Reads all user profiles' URLs from the users who have reviewed fishing tourism businesses
URLs = read_comments_from_files()['reviewer_profile']

# Reads a URL at a time and calls the scraping function
# for url in tqdm(reversed(URLs)): # inverse order
for url in tqdm(URLs[796:]): # correct order
    try:
        driver.set_page_load_timeout(10)
        get_user_profile_by_url(url)
    except TimeoutException as e:
        isrunning = 0
        print('\nThere is an issue, check again ' + url + " & Exception: " + str(e))
        driver.close()

    print()

print('\nProgram is complete.')
driver.close()
