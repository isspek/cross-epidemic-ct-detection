import numpy as np
import pandas as pd
import os
import random
import torch
from argparse import ArgumentParser
from docarray import Document, DocumentArray
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
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


def index(collection_name, chunk_size, search_collection, clear):
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

    # da = DocumentArray(
    #     storage='redis',
    #     config={'host': 'localhost', 'port': 6379, 'n_dim': int(SENTENCE_EMBEDDING_DIM),
    #             'index_name': search_collection, 'index_text': True},
    #
    # )

    if clear:
        da.clear()

    db, client = connect_to_db()
    collection = db[collection_name]

    count = collection.count_documents({})

    logger.info(f'Number of the documents to be embeded: {count}')

    # Initialize Sentence-BERT model
    model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

    documents = collection.find({}, batch_size=chunk_size)

    ops = []
    for document in tqdm(documents, total=count):
        ops.append({
            'cleaned_text': document['cleaned_text'],
            'id': str(document['id_str']) if 'ebola' not in collection_name else str(document['id'])
        })

        if len(ops) > chunk_size:
            batch_embeddings = model.encode(list(map(lambda x: x['cleaned_text'], ops
                                                     )), show_progress_bar=False)
            batch_embeddings = np.split(batch_embeddings, len(ops))

            for idx, batch_embedding in enumerate(batch_embeddings):
                ops[idx]["embedding"] = batch_embedding.tolist()[0]

            with da:
                da.extend(list(map(lambda x: Document(
                    text=x['cleaned_text'],
                    id=x['id'],
                    embedding=x['embedding']), ops)),
                          thread_count=4,
                          chunk_size=500,
                          max_chunk_bytes=104857600,
                          queue_size=4,
                          )

            ops = []

    if len(ops) > 0:
        batch_embeddings = model.encode(ops, show_progress_bar=False)
        batch_embeddings = np.split(batch_embeddings, len(ops))

        for idx, batch_embedding in enumerate(batch_embeddings):
            ops[idx]["embedding"] = batch_embedding.tolist()[0]

        with da:
            da.extend(list(map(lambda x: Document(
                text=x['cleaned_text'],
                embedding=x['embedding']), ops)),
                          thread_count=4,
                          chunk_size=500,
                          max_chunk_bytes=104857600,
                          queue_size=4)

    client.close()


def export_embeddings(input_file, search_collection, output_path):
    da = DocumentArray(
        storage='redis',
        config={'host': 'localhost', 'port': 6379, 'n_dim': int(SENTENCE_EMBEDDING_DIM),
                'index_name': search_collection, 'index_text': True},

    )

    logger.info(da.summary)

    data = pd.read_csv(input_file, sep='\t', lineterminator='\n')

    embeddings = []
    for idx, row in tqdm(data.iterrows()):
        cleaned_text = row['cleaned_text']
        id_str = str(row['id_str'])
        try:
            embeddings.append(
                {
                    'embedding': da[id_str].embedding,
                    'id': id_str,
                    'cleaned_text': cleaned_text
                })
        except Exception as e:
            logger.error(e)
            pass

    logger.info(f'Num of embeddings {len(embeddings)}')

    with open(output_path, "w") as f:
        f.write(pd.DataFrame(embeddings).to_json(orient='records', lines=True, force_ascii=False))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda_device', type=str)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--collection', type=str)
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--search_collection', type=str)
    parser.add_argument('--embedding_name', type=str)
    parser.add_argument('--index', action='store_true')
    parser.add_argument('--export', action='store_true')
    parser.add_argument('--clear', action='store_true')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    set_random_seed(args.random_seed)

    if args.index:
        index(collection_name=args.collection, chunk_size=args.chunk_size,
              search_collection=args.search_collection, clear=args.clear)

    if args.export:
        export_embeddings(input_file=args.input_file, search_collection=args.search_collection,
                          output_path=args.output_path)
