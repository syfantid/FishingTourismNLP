# Code by https://github.com/Sanjanaekanayake/tripadvisor-user-profiles-scrapper


from selenium import webdriver
import csv
import requests
import re
import time

from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager

from data_annotation.annotation_file_extractor import read_comments_from_files

# import the webdriver, chrome driver is recommended
driver = webdriver.Chrome(ChromeDriverManager().install())

driver.set_page_load_timeout(2)
filename = ""
maxcount = 100
i = 0


# function to check if the button is on the page, to avoid miss-click problem
def check_exists_by_xpath(xpath):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
    time.sleep(2)


def writecsv(c1, c2, c3, c4, c5):
    with open(filename, mode='a', newline='') as f:
        # keys = ['name', 'age', 'job', 'city']
        writer = csv.writer(f)
        writer.writerow([str(c1), str(c2), str(c3), str(c4), str(c5)])


# In[329]:

def get_user_reviews(URL, endcount):
    revews_tab = driver.find_element_by_xpath('//a[@data-tab-name="Reviews"]')
    revews_tab.click()
    time.sleep(2)

    # if (check_exists_by_xpath("//div[@id='content']")):
    if check_exists_by_xpath("//span[@class='_1ogwMK0l']"):
        # to expand the review 
        show_more_button = driver.find_element_by_xpath("//span[@class='_1ogwMK0l']")
        show_more_button.click()
        time.sleep(30)

    while driver.find_elements_by_xpath("//div[@style='position:relative']/div"):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        review = driver.find_elements_by_xpath("//div[@style='position:relative']/div")
        element_count = len(review)
        if (element_count is endcount):
            print('end')
            break
        else:
            continue

    for j in range(element_count):

        try:

            name = review[j].find_element_by_xpath(".//div[contains(@class, '_2fxQ4TOx')]").text
            review_title = review[j].find_element_by_xpath(".//div[contains(@class, '_3IEJ3tAK _2K4zZcBv')]").text
            review_date = review[j].find_element_by_xpath(".//div[contains(@class, '_3Coh9OJA')]").text
            review_for = review[j].find_element_by_xpath(".//div[contains(@class, '_2ys8zX0p ui_link')]").text
            review_summary = review[j].find_element_by_xpath(".//div[contains(@class, '_1kKLd-3D')]/a").get_attribute(
                "href")

            driver.execute_script("window.open('');")
            driver.switch_to.window(driver.window_handles[1])
            driver.get(review_summary)
            time.sleep(2)

            if (check_exists_by_xpath("//span[@class='fullText hidden']")):
                read_more = driver.find_elements_by_xpath("//div[@class='reviewSelector']/div/div[2]/div[3]/div/p/span")
                read_more[2].click()
                review_text = read_more[1].text
            elif (check_exists_by_xpath("//span[@class='fullText ']")):
                read_more = driver.find_elements_by_xpath("//div[@class='reviewSelector']/div/div[2]/div[3]/div/p/span")
                review_text = read_more[0].text
            else:
                reviewdetails = driver.find_elements_by_xpath(
                    "//div[@class='reviewSelector']/div/div[2]/div/div/div[3]/div/p")
                review_text = reviewdetails[0].text

            driver.close()
            driver.switch_to.window(driver.window_handles[0])
            writecsv(name, review_title, review_text, review_date, review_for)



        except:
            print('review not found')
            break



def scrape_user_profile(url):
    # get the name of place for csv file name
    global filename

    driver.get(url)
    driver.maximize_window()

    # get element count
    count = driver.find_element_by_class_name("_1q4H5LOk").text.replace(",", "")

    # username as the filename
    username = driver.find_element_by_class_name("gf69u3Nd").text

    filename = username + ".csv"

    # open csv file and add titles
    with open(filename, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(
            [str('user name'), str('reviewTitle'), str('reviewDetails'), str('reviewDate'), str('reviewFor')])

    endcount = int(maxcount) if int(count) > int(maxcount) else int(count)

    get_user_reviews(url, endcount)
    print('save reviews in page = ', str(endcount), ' user = ', filename)
    print()


def scrape_user_profiles():
    """
    Function to scrape the profiles of all the users who have reviewed an enterprise of interest
    :return: None
    """
    profile_urls = read_comments_from_files()['reviewer_profile']
    for profile_url in profile_urls:
        try:
            scrape_user_profile(profile_url)
        except:
            print('There is an issue, check again ' + profile_url)
    print('All user profiles are now scraped.')
    driver.close()


scrape_user_profiles()
