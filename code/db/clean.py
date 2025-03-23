import re
import pymongo
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from code.db.utils import connect_to_db, COLLECTION_TEXT_MAPPING, STOP_WORDS

logger.add(f"{__name__}.log", rotation="500 MB")


def remove_duplicates(collection_name, chunk_size):
    db, client = connect_to_db()
    collection = db[collection_name]

    '''
    The pipeline first group the documents by full_text and count the occurences. Match shows the full_texts whose occurences are larger than 1
    '''
    pipeline = [{
        "$group": {
            "_id": f"${COLLECTION_TEXT_MAPPING[collection_name]}",
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }
    }, {
        "$match": {
            "_id": {"$ne": None},
            "count": {"$gt": 1}
        }
    }]

    groups = collection.aggregate(pipeline, allowDiskUse=True)

    ids_be_removed = []
    for group in groups:
        ids_be_removed.extend(group['ids'][1:])
        if len(ids_be_removed) > chunk_size:
            collection.delete_many({"_id": {"$in": ids_be_removed}})
            logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")
            ids_be_removed = []

    if len(ids_be_removed) > 0:
        collection.delete_many({"_id": {"$in": ids_be_removed}})
        logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")

    logger.info(
        f"After removing the duplicates the number of the samples of {collection_name}: {collection.count_documents({})}")

    client.close()


def keep_only_x_lang(collection_name, chunk_size, language):
    db, client = connect_to_db()
    collection = db[collection_name]

    '''
    The query finds the samples in non-x language
    '''
    samples = collection.find({"lang": {"$ne": language}}, batch_size=chunk_size)

    ids_be_removed = []
    for sample in samples:
        ids_be_removed.append(sample['_id'])

        if len(ids_be_removed) > chunk_size:
            collection.delete_many({"_id": {"$in": ids_be_removed}})
            logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")
            ids_be_removed = []

    if len(ids_be_removed) > 0:
        collection.delete_many({"_id": {"$in": ids_be_removed}})
        logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")

    logger.info(
        f"After removing the non-english samples, the number of the samples of {collection_name}: {collection.count_documents({})}")

    client.close()


def remove_by_field(collection_name, chunk_size, field):
    db, client = connect_to_db()
    collection = db[collection_name]

    '''
    The pipeline first group the documents by cleaned_texts and count the occurences. Match shows the full_texts whose occurences are larger than 1
    '''
    pipeline = [{
        "$group": {
            "_id": f"${field}",
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }
    }, {
        "$match": {
            "_id": {"$ne": None},
            "count": {"$gt": 1}
        }
    }]

    groups = collection.aggregate(pipeline, allowDiskUse=True)

    ids_be_removed = []
    for group in groups:
        ids_be_removed.extend(group['ids'][1:])
        if len(ids_be_removed) > chunk_size:
            collection.delete_many({"_id": {"$in": ids_be_removed}})
            logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")
            ids_be_removed = []

    if len(ids_be_removed) > 0:
        collection.delete_many({"_id": {"$in": ids_be_removed}})
        logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")

    logger.info(
        f"After removing the duplicates the number of the samples of {collection_name}: {collection.count_documents({})}")

    client.close()

def remove_by_user_posts(collection_name, chunk_size, count):
    db, client = connect_to_db()
    collection = db[collection_name]

    '''
    The pipeline first group the documents by cleaned_texts and count the occurences. Match shows the full_texts whose occurences are larger than 1
    '''
    pipeline = [{
        "$group": {
            "_id": f"$user_screen_name" if 'ebola' in collection_name else f"$user.name",
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }
    }, {
        "$match": {
            "_id": {"$ne": None},
            "count": {"$lt": count}
        }
    }]

    groups = collection.aggregate(pipeline, allowDiskUse=True)

    ids_be_removed = []
    for group in groups:
        ids_be_removed.extend(group['ids'])
        if len(ids_be_removed) > chunk_size:
            collection.delete_many({"_id": {"$in": ids_be_removed}})
            logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")
            ids_be_removed = []

    if len(ids_be_removed) > 0:
        collection.delete_many({"_id": {"$in": ids_be_removed}})
        logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")

    logger.info(
        f"After removing the duplicates the number of the samples of {collection_name}: {collection.count_documents({})}")

    client.close()


def remove_by_frequency(collection_name, chunk_size, limit):
    db, client = connect_to_db()
    collection = db[collection_name]

    samples = collection.find({}, batch_size=chunk_size)

    ids_be_removed = []

    for sample in tqdm(samples):
        text = sample['cleaned_text']

        if len(text.split()) <= limit:
            ids_be_removed.append(sample['_id'])

        if len(ids_be_removed) > chunk_size:
            collection.delete_many({"_id": {"$in": ids_be_removed}})
            logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")
            ids_be_removed = []

    if len(ids_be_removed) > 0:
        collection.delete_many({"_id": {"$in": ids_be_removed}})
        logger.info(f"Removed {len(ids_be_removed)} from {collection_name}")

    logger.info(
        f"After removing the duplicates the number of the samples of {collection_name}: {collection.count_documents({})}")

    client.close()


def clean_tweet(tweet, keep_numbers:False):
    # lower the tweets
    tweet = tweet.lower()

    # remove special tokens
    tweet = re.sub(r'(?:\b|\s)amp(?:\b|\s)', '', tweet)
    tweet = re.sub(r'(?:\b|\s)via(?:\b|\s)', '', tweet)
    tweet = re.sub(r'^rt[\s]+', '', tweet)

    # Remove mentions and handles (@usernames)
    tweet = re.sub(r'@[\w\d]+', '', tweet)

    # Remove hashtags
    tweet = re.sub(r'#[\w\d]+', '', tweet)

    # Remove URLs
    tweet = re.sub(r'https?://\S+', '', tweet)

    # Remove non-alphanumeric character sequences
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Remove dates in the format of YYYY-MM-DD or YYYY/MM/DD
    tweet = re.sub(r'\b(?:20|19)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b', '', tweet)

    # Remove numbers
    if not keep_numbers:
        tweet = re.sub(r'\b\d+\b', '', tweet)

    # Remove punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Replace multiple whitespaces with a single whitespace
    tweet = re.sub(r'\s+', ' ', tweet)

    # Remove stop words
    tweet_tokens = [word.strip() for word in tweet.split() if not word.lower() in STOP_WORDS or len(word.strip())>=2]

    # Join the remaining tokens back into a single string
    tweet = ' '.join(tweet_tokens)

    # Remove leading and trailing whitespaces
    tweet = tweet.strip()

    return tweet

def clean_tweet_v2(tweet, keep_numbers:False):
    # lower the tweets
    tweet = tweet.lower()

    # remove special tokens
    tweet = re.sub(r'(?:\b|\s)amp(?:\b|\s)', '', tweet)
    tweet = re.sub(r'(?:\b|\s)via(?:\b|\s)', '', tweet)
    tweet = re.sub(r'^rt[\s]+', '', tweet)

    # Remove mentions and handles (@usernames)
    tweet = re.sub(r'@[\w\d]+', '', tweet)

    # Remove hashtags
    tweet = re.sub(r'#[\w\d]+', '', tweet)

    # Remove URLs
    tweet = re.sub(r'https?://\S+', '', tweet)

    # Remove non-alphanumeric character sequences
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Remove dates in the format of YYYY-MM-DD or YYYY/MM/DD
    tweet = re.sub(r'\b(?:20|19)\d{2}[-/](?:0[1-9]|1[0-2])[-/](?:0[1-9]|[12]\d|3[01])\b', '', tweet)

    # Remove numbers
    if not keep_numbers:
        tweet = re.sub(r'\b\d+\b', '', tweet)

    # Remove punctuations
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Replace multiple whitespaces with a single whitespace
    tweet = re.sub(r'\s+', ' ', tweet)

    # Remove stop words
    tweet_tokens = [word.strip() for word in tweet.split() if not word.lower() in STOP_WORDS or len(word.strip())>=2]

    # Join the remaining tokens back into a single string
    tweet = ' '.join(tweet_tokens)

    # Remove leading and trailing whitespaces
    tweet = tweet.strip()
    
    for i in ["ebola", "zika", "covid", "covid-19", "corona", "monkeypox", "monkey pox"]:
        case_sens = re.compile(re.escape(i), re.IGNORECASE)
        tweet=case_sens.sub('disease', tweet)

    return tweet


def clean_tweets(input_collection, chunk_size, force):
    db, client = connect_to_db()
    _input_collection = db[input_collection]
    if not force:
        count = _input_collection.count_documents({"cleaned_text": {"$exists": False}})
        samples = _input_collection.find({"cleaned_text": {"$exists": False}}, batch_size=chunk_size)

    else:
        count = _input_collection.count_documents({})
        samples = _input_collection.find({}, batch_size=chunk_size)
    logger.info(f"{count} documents have not been cleaned yet.")

    ops = []
    for sample in tqdm(samples, total=count):
        cleaned_text = clean_tweet(sample["full_text"]) if 'full_text' in sample else clean_tweet(sample["text"])
        ops.append(pymongo.UpdateOne({"_id": sample["_id"]}, {"$set": {"cleaned_text": cleaned_text}}))

        if len(ops) > chunk_size:
            _input_collection.bulk_write(ops)
            logger.info(f"sample {len(ops)} is inserted.")
            ops = []

    if len(ops) > 0:
        _input_collection.bulk_write(ops)
        logger.info(f"sample {len(ops)} is inserted.")

    client.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_collection', help="input collection name")
    parser.add_argument('--output_collection',
                        help="output collection name that the cleaning operations will be saved.")
    parser.add_argument('--chunksize', help="set up chunksize for the large datasets", type=int)
    parser.add_argument('--remove_duplicates',
                        help="Remove tweets that contain exact same texts and keep only one sample",
                        action='store_true')
    parser.add_argument('--remove_near_duplicates',
                        help="Remove tweets that are almost same but have a slight difference (e.g. a punctuation difference) and keep the normalized sample",
                        action='store_true')

    parser.add_argument('--keep_only_x_lang',
                        help="Keep only tweets in x language. X refers to the language specified by an user",
                        type=str)

    parser.add_argument('--clean_tweets',
                        help="Apply preprocessing on tweets, and assign to a field called cleaned_text",
                        action='store_true')

    parser.add_argument('--field',
                        help="Apply preprocessing on tweets, and assign to a field called cleaned_text",
                        type=str)

    parser.add_argument('--remove_by_field',
                        help="Remove tweets group by FIELD",
                        action='store_true')
    parser.add_argument('--force',
                        help="force to do the operation",
                        action='store_true')
    parser.add_argument('--remove_by_frequency',
                        help="Remove tweets whose word number is smaller than a limit that is defined by a user",
                        action='store_true')
    parser.add_argument('--remove_by_user_posts',
                        help="Remove tweets whose word number is smaller than a limit that is defined by a user",
                        action='store_true')
    parser.add_argument('--count',
                        type=int)

    args = parser.parse_args()

    if args.remove_duplicates:
        remove_duplicates(collection_name=args.input_collection, chunk_size=args.chunksize)

    if args.keep_only_x_lang:
        keep_only_x_lang(collection_name=args.input_collection, chunk_size=args.chunksize,
                         language=args.keep_only_x_lang)

    if args.clean_tweets:
        clean_tweets(input_collection=args.input_collection, chunk_size=args.chunksize, force=args.force)

    if args.remove_by_field:
        remove_by_field(collection_name=args.input_collection, chunk_size=args.chunksize, field=args.field)

    if args.remove_by_frequency:
        remove_by_frequency(collection_name=args.input_collection, chunk_size=args.chunksize, limit=2)

    if args.remove_by_user_posts:
        remove_by_user_posts(collection_name=args.input_collection, chunk_size=args.chunksize, count=args.count)
