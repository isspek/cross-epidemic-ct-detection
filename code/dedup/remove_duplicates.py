import _pickle
from argparse import ArgumentParser
from loguru import logger
from code.db.utils import connect_to_db
from pathlib import Path

logger.add(f"{__name__}.log", rotation="500 MB")

def load_result(collection='ebola', store_folder='./data/candidates'):
    '''
    Load the result of the deduplication from the collection,
    :param collection: ebola, zika, or monkeypox
    :param store_folder: root folder where per-collection datasets are stored
    :return: list of IDs of duplicates to be removed from the collection
    '''
    store_folder = Path(store_folder) / collection
    out_file = Path(store_folder) / 'result.pickle'
    if out_file.exists(): rep_ids, dup_ids =  _pickle.load(open(out_file, 'rb'))
    else: raise ValueError(f'No results stores in {str(store_folder)}')
    dup_list = [did for did_lst in dup_ids.values() for did in did_lst]
    return dup_list
def remove_deduplicates(input_collection):
    # dedup_list = load_result(collection=input_collection.split('_')[0])
    dedup_list = load_result(collection=input_collection)

    logger.info(f'Number documents that needs to be removed: {len(dedup_list)}')
    db, client = connect_to_db()
    collection = db[input_collection]

    count = collection.count_documents({})

    logger.info(f'Number of documents {count}')

    result = collection.delete_many({"id_str":  {"$in": list(map(lambda x:int(x),dedup_list))}})

    logger.info(f"{result.deleted_count} documents were deleted.")


    client.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_collection', help="input collection name")
    parser.add_argument('--chunksize', help="set up chunksize for the large datasets", type=int)

    args = parser.parse_args()

    remove_deduplicates(input_collection=args.input_collection)