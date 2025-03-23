import numpy as np
import pymongo
import os
import random
import torch
import pandas as pd
from argparse import ArgumentParser
from docarray import Document, DocumentArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pandas as pd
from loguru import logger
from code.db.utils import connect_to_db
from code.db.utils import SENTENCE_EMBEDDING_MODEL, SENTENCE_EMBEDDING_DIM, ES_HOST

logger.add(f"{__name__}.log", rotation="500 MB")


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def search(collection_name, chunk_size, search_collection, claims_file, output_file, limit):
    # da = DocumentArray(
    #     storage='redis',
    #     config={'host': 'localhost', 'port': 6379, 'n_dim': int(SENTENCE_EMBEDDING_DIM),
    #             'index_name': search_collection, 'index_text': True},
    #
    # )
    da = DocumentArray(
        storage='elasticsearch',
        config={
            'hosts': ES_HOST,
            'n_dim': int(SENTENCE_EMBEDDING_DIM),
            'index_name': search_collection,
            'es_config': {'request_timeout': 120},
            'index_text': True
        },
    )

    db, client = connect_to_db()
    collection = db[collection_name]

    # Initialize Sentence-BERT model
    model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

    claims_file = pd.read_csv(claims_file, sep='\t')

    # queries = claims_file['ext-summary'].unique()
    queries = claims_file['cleaned_text'].unique()

    search_dataset = []
    for query in tqdm(queries, total=len(queries)):

        try:
            results = da.find(model.encode(query), metric='cosine', limit=limit)
            for result in results:
                sample = collection.find_one({"cleaned_text": result.text}, batch_size=chunk_size)
                # print(sample)
                search_dataset.append({
                    'cleaned_text': result.text,
                    'claim': query,
                    'created_at': sample['created_at'],
                    'full_text': sample["text"] if 'ebola' in collection_name else sample["full_text"],
                    'user_name': sample["user_name"] if 'ebola' in collection_name else sample["user"]["screen_name"],
                    'id_str': sample["id"] if 'ebola' in collection_name else sample["id_str"]
                })
        except Exception as e:
            logger.error(e)

    new_data = pd.DataFrame(search_dataset)

    logger.info(f'Number of the documents {len(new_data)}')

    new_data.drop_duplicates(subset=['cleaned_text'], keep='first', inplace=True)

    logger.info(f'Number of the documents after dropping {len(new_data)}')

    new_data.to_csv(output_file, sep='\t', index=False)

    client.close()


def keyword_search(collection_name, keywords, output_file):
    db, client = connect_to_db()
    collection = db[collection_name]

    keywords = pd.read_csv(keywords, sep='\t')
    keywords = keywords["0"].tolist()

    search_dataset = []
    for keyword in tqdm(keywords, total=len(keywords)):
        query = {'cleaned_text': {'$regex': keyword, '$options': 'i'}}
        results = collection.find(query)

        for result in results:
            search_dataset.append({
                'cleaned_text': result["cleaned_text"],
                'keyword': keyword,
                'created_at': result['created_at'],
                'full_text': result["full_text"],
                'user_name': result["user_name"] if 'ebola' in collection_name else result["user"]["screen_name"],
                'id_str': result["id"] if 'ebola' in collection_name else result["id_str"]
            })

    search_dataset = pd.DataFrame(search_dataset)
    search_dataset.drop_duplicates(subset=['cleaned_text'], keep='first', inplace=True)

    logger.info(f"Number of documents {len(search_dataset)}")

    search_dataset.to_csv(output_file, sep='\t', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda_device', type=str)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--collection', type=str)
    parser.add_argument('--search_collection', type=str)
    parser.add_argument('--claims_file', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--keywords', type=str)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--keyword_search', action='store_true')

    args = parser.parse_args()
    set_random_seed(args.random_seed)

    if args.keyword_search:
        keyword_search(collection_name=args.collection,
                       keywords=args.keywords,
                       output_file=args.output_file)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
        search(collection_name=args.collection,
               chunk_size=args.chunk_size,
               search_collection=args.search_collection,
               claims_file=args.claims_file,
               output_file=args.output_file,
               limit=args.limit)
