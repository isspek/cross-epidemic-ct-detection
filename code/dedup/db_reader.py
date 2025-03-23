from tqdm import tqdm
from loguru import logger
from code.db.utils import connect_to_db

logger.add(f"{__name__}.log", rotation="500 MB")


def iterate_samples(input_collection, chunk_size=1000):
    db, client = connect_to_db()
    _input_collection = db[input_collection]

    count = _input_collection.count_documents({"levenstein": {"$exists": False}})
    # count = _input_collection.count_documents({})

    logger.info(f"{count} documents have not been cleaned yet.")

    samples = _input_collection.find({"levenstein": {"$exists": False}}, batch_size=chunk_size)
    # samples = _input_collection.find({}, batch_size=chunk_size)

    for sample in tqdm(samples, total=count):
        id_str = sample["id_str"]
        text = sample["cleaned_text"]

        ## if you want to write the processed field to the db, adopt the below commented code
        # ops.append(pymongo.UpdateOne(sample, {"$set": {"levenstein": processed_text}}))

        # if len(ops) > chunk_size:
        #     _input_collection.bulk_write(ops)
        #     logger.info(f"sample {len(ops)} is inserted.")
        #     ops = []

        yield (id_str, text)

    # if len(ops) > 0:
    #     _input_collection.bulk_write(ops)
    #     logger.info(f"sample {len(ops)} is inserted.")

    client.close()

def test_collection_access(collection):
    cnt = 0
    for id, txt in iterate_samples(collection):
        print(f'ID: {id}; TEXT: [{txt}]')
        cnt += 1
    print(f'TOTAL TEXTS: {cnt}')

if __name__ == '__main__':
    test_collection_access('zika_with_urls_candidates')
