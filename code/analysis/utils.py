import re

STOP_WORDS = ['&amp']
DISEASE_TAG = '[DISEASE]'


def clean_tweet_v2(tweet, keep_numbers: bool = False, lowerize: bool = False, keep_punctuation: bool = False, replace_disease_names: bool = False, remove_disease_name: bool = False):
    if lowerize:
        # lower the tweets
        tweet = tweet.lower()

    # Remove mentions and handles (@usernames)
    tweet = re.sub(r'@[\w\d]+', '', tweet)

    # Remove hashtags
    tweet = re.sub(r'#[\w\d]+', '', tweet)

    # Remove URLs
    tweet = re.sub(r'https?://\S+', '', tweet)

    # # Remove non-alphanumeric character sequences
    # tweet = re.sub(r'[^\w\s]', '', tweet)

    # Remove retweet handles
    tweet = re.sub(r'^rt\s+', '', tweet, flags=re.IGNORECASE)

    # Remove dates in the format of YYYY-MM-DD or YYYY/MM/DD
    tweet = re.sub(r'\b(?:20|19)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b', '', tweet)

    # Remove numbers
    if not keep_numbers:
        tweet = re.sub(r'\b\d+\b', '', tweet)

    # Remove punctuations
    if not keep_punctuation:
        tweet = re.sub(r'[^\w\s]', '', tweet)

    # Replace multiple whitespaces with a single whitespace
    tweet = re.sub(r'\s+', ' ', tweet)

    # Remove leading and trailing whitespaces
    tweet = tweet.strip()

    if replace_disease_names:
        for i in ["ebola", "zika", "zica", "covid", "covid-19", "corona", "monkeypox", "monkey pox", "covid19", "coronavirus", "corona 19",
                  "covid 19", "COVID19", "COVID", "COVID-19", "SARS-CoV", "sars-cov", "Coronavirus"]:
            case_sens = re.compile(re.escape(i), re.IGNORECASE)
            tweet = case_sens.sub(DISEASE_TAG, tweet)
        tweet = tweet.replace(f'{DISEASE_TAG}19', DISEASE_TAG)
        tweet = tweet.replace(f'{DISEASE_TAG}virus', DISEASE_TAG)

        if remove_disease_name:
            tweet = tweet.replace(DISEASE_TAG, '')

    for i in STOP_WORDS:
        case_sens = re.compile(re.escape(i), re.IGNORECASE)
        tweet = case_sens.sub('', tweet)

    return tweet
