import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_HOST = os.getenv("MONGO_HOST")
MONGO_PORT = os.getenv("MONGO_PORT")
MONGO_UNAME = os.getenv("UNAME")
MONGO_PASS = os.getenv("PASS")
MONGO_DB_URL = f"mongodb://{MONGO_UNAME}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}"

SENTENCE_EMBEDDING_MODEL = os.getenv("SENTENCE_EMBEDDING_MODEL")
SENTENCE_EMBEDDING_DIM = os.getenv("SENTENCE_EMBEDDING_DIM")

ES_HOST = os.getenv("ES_HOST")
ES_PORT = os.getenv("ES_PORT")

STOP_WORDS = ['via', 'amp', 'http', 'cst', 'cest', 'am', 'pm', 'rt']

def fetch_batch(self, query={}, chunk_size=100):
    '''
    Adapted from https://pastebin.com/DihddttJ
    :param self:
    :param query:
    :param chunk_size:
    :return:
    '''
    # 2000
    total_count = self.cursor.count_documents(filter=query)

    # 2000//100
    total_pages = total_count // chunk_size

    page_size = chunk_size

    if total_count % chunk_size != 0:
        total_pages += 1

    for page_number in range(1, total_pages + 1):
        skips = page_size * (page_number - 1)
        cursor = self.cursor.find(query).skip(skips).limit(page_size)
        yield [x for x in cursor]

def connect_to_db():
    client = MongoClient(MONGO_DB_URL)
    assert client.server_info()
    db = client.conspiracy
    return db, client

COLLECTION_TEXT_MAPPING = {
    "ebola": "text",
    "zika": "full_text",
    "zika_with_urls": "full_text",
    "monkeypox": "full_text",
    "covid_hoax_and_hidden_agenda": "text",
    "covid_vax_misinfo": "text",
    "covid_media_eval": "tweet_text"
}
