from textblob import TextBlob
from nltk.tag import pos_tag
from wnaffect.wnaffect import WNAffect
from nltk.tokenize import word_tokenize
from collections import Counter


def get_review_sentiment(text):
    sentiment = TextBlob(text).polarity
    return sentiment

def get_all_sentiments(texts):
    #returns all positive and negative sentiments for english text
    positive_sentiment = []
    negative_sentiment = []
    for t in texts:
        try:
            sentiment = get_review_sentiment(t)
            if(sentiment >0):
                positive_sentiment.append(sentiment)
            elif(sentiment<0):
                negative_sentiment.append(sentiment)
        except TypeError:
            print ('blank post- non-english post')
        continue

    return positive_sentiment, negative_sentiment

def get_review_emotion(text):
    # Detect emotions based on wna
    emotions = []
    wna = WNAffect('wnaffect/wordnet-1.6/', 'wnaffect/wn-domains-3.2/')
    primary_emotions = ['anger', 'disgust', 'negative-fear', 'joy', 'sadness', 'surprise', 'positive-emotion',
                        'negative-emotion']

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    for word, pos in tagged:
        emo = wna.get_emotion(word, pos)
        emo_name = str(emo)
        if (emo_name != "None"):
            while (not (any(emo_name in x for x in primary_emotions))):
                emo = emo.get_level(emo.level - 1)
                emo_name = str(emo)
            emotions.append(emo_name)

    return emotions


def get_all_emotions(texts):
    #returns the occurences of the basic affects

    allemotions=[]

    for t in texts:
        try:
            emotions = get_review_emotion(t)
            allemotions.append(emotions)
        except TypeError:
            print ('blank post')
        continue

    return allemotions


def get_emotions_from_emojis(text):
    #returns the occurences of affective emojis

    def find_affective_emoji(affect_list, text):
        emojis = []
        for emo in affect_list:
            if type(text) is str:
                if emo in text:
                    emojis.append(emo)
        return emojis


    # affective emojis
    joy_emojis_f = ['\U0001f600', '\U0001f602', '\U0001f603', '\U0001f604', '\U0001f606', '\U0001f607', '\U0001f609',
                     '\U0001f60a', '\U0001f60e', '\U0001f60f', '\U0001f31e',  '\U0001f60b', '\U0001f60c',
                     '\U0001f60d']
    joy_emojis_s = ['\U0001f618', '\U0001f61c', '\U0001f61d', '\U0001f61b', '\U0001f63a', '\U0001f638', '\U0001f639',
                     '\U0001f63b', '\U0001f63c', '\U0001f496', '\U0001f495', '\U0001f601']
    anger_emojis = ['\U0001f62c', '\U0001f620', '\U0001f610', '\U0001f611', '\U0001f620', '\U0001f621', '\U0001f616',
                     '\U0001f624', '\U0001f63e']
    disgust_emojis = ['\U0001f4a9']
    fear_emojis = ['\U0001f605', '\U0001f626', '\U0001f627', '\U0001f631', '\U0001f628', '\U0001f630', '\U0001f640']
    sad_emojis = ['\U0001f614', '\U0001f615', '\U0001f62b', '\U0001f629', '\U0001f622', '\U0001f625', '\U0001f62a',
                  '\U0001f613', '\U0001f62d', '\U0001f63f', '\U0001f494']
    surprise_emojis = ['\U0001f633', '\U0001f62f', '\U0001f635', '\U0001f632']

    #initialize lists
    joy_emojis_f_ = []
    joy_emojis_s_ = []
    anger_emojis_ = []
    disgust_emojis_ = []
    fear_emojis_ = []
    sad_emojis_ = []
    surprise_emojis_ = []

    try:
        # find joy emojis
        joy_emojis_f_.append(find_affective_emoji(joy_emojis_f, text))
        joy_emojis_s_.append(find_affective_emoji(joy_emojis_s, text))
        joy_emojis_ = joy_emojis_f_ + joy_emojis_s_
        joy = [val for sublist in list(filter(None, joy_emojis_)) for val in sublist]

        # find anger emojis
        anger_emojis_.append(find_affective_emoji(anger_emojis, text))
        anger = [val for sublist in list(filter(None, anger_emojis_)) for val in sublist]

        # find disgust emojis
        disgust_emojis_.append(find_affective_emoji(disgust_emojis, text))
        disgust = [val for sublist in list(filter(None, disgust_emojis_)) for val in sublist]

        # find fear emojis
        fear_emojis_.append(find_affective_emoji(fear_emojis, text))
        fear = [val for sublist in list(filter(None, fear_emojis_)) for val in sublist]

        # find sad emojis
        sad_emojis_.append(find_affective_emoji(sad_emojis, text))
        sadness = [val for sublist in list(filter(None, sad_emojis_)) for val in sublist]

        # find surprise emojis
        surprise_emojis_.append(find_affective_emoji(surprise_emojis, text))
        surprise = [val for sublist in list(filter(None, surprise_emojis_)) for val in sublist]


    except TypeError:
        print('blank post')

    joy = Counter(joy).most_common()
    anger = Counter(anger).most_common()
    disgust = Counter(disgust).most_common()
    fear = Counter(fear).most_common()
    sadness = Counter(sadness).most_common()
    surprise = Counter(surprise).most_common()

    emo = []

    return emo.extend((joy, anger, disgust, fear, sadness, surprise))


def get_all_emoji_emotions(texts):
    # returns the occurences of the basic affects

    allemojiemotions = []

    for t in texts:
        try:
            joy, anger, disgust, fear, sadness, surprise = get_emotions_from_emojis(t)
            allemojiemotions.append(joy)
            allemojiemotions.append(anger)
            allemojiemotions.append(disgust)
            allemojiemotions.append(fear)
            allemojiemotions.append(sadness)
            allemojiemotions.append(surprise)
        except TypeError:
            print('blank post')
        continue

    return  allemojiemotions

# df['text_sentiment'] = df['text'].apply(lambda x: get_review_sentiment(x))
# df['text_emotion'] = df['text'].apply(lambda x: get_review_emotion(x))
# df['text_emojis_emotion'] = df['text'].apply(lambda x: get_emotions_from_emojis(x)) #todo DIMITRA - does not identify the emojis






# all_positive_sentiment, all_negative_sentiment = get_all_sentiments()
# print("Total positive sentiment" + str(sum(all_positive_sentiment)) + " of " + str(len(all_positive_sentiment)) + " english reviews...")
# print("Total negative sentiment" + str(sum(all_negative_sentiment)) + " of " + str(len(all_negative_sentiment)) + " english reviews...")
#
# allemotions = get_all_emotions(col)
# print("Emotions found.... ", Counter(allemotions))
#
# joy, anger, disgust, fear, sadness, surprise = get_emotions_from_emojis(col)
#
# print("Joy found in ", "{:.1%}".format(len(joy)/(len(joy)+len(anger)+len(disgust)+(len(fear)+len(sadness)+len(surprise)))))
# joy = Counter(joy).most_common()
# for emoji, count in joy:
#     print(emoji, count)
#
# print("Anger found in ", "{:.1%}".format(len(anger)/(len(joy)+len(anger)+len(disgust)+(len(fear)+len(sadness)+len(surprise)))))
# anger = Counter(anger).most_common()
# for emoji, count in anger:
#     print(emoji, count)
#
# print("Disgust found in ", "{:.1%}".format(len(disgust)/(len(joy)+len(anger)+len(disgust)+(len(fear)+len(sadness)+len(surprise)))))
# disgust = Counter(disgust).most_common()
# for emoji, count in disgust:
#     print(emoji, count)
#
# print("Fear found in ", "{:.1%}".format(len(fear)/(len(joy)+len(anger)+len(disgust)+(len(fear)+len(sadness)+len(surprise)))))
# fear = Counter(fear).most_common()
# for emoji, count in fear:
#     print(emoji, count)
#
# print("Sadness found in ", "{:.1%}".format(len(sadness)/(len(joy)+len(anger)+len(disgust)+(len(fear)+len(sadness)+len(surprise)))))
# sadness = Counter(sadness).most_common()
# for emoji, count in sadness:
#     print(emoji, count)
#
# print("Surprise found in ", "{:.1%}".format(len(surprise)/(len(joy)+len(anger)+len(disgust)+(len(fear)+len(sadness)+len(surprise)))))
# surprise = Counter(surprise).most_common()
# for emoji, count in surprise:
#     print(emoji, count)
#

