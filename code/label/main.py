import pymongo
import tldextract
from tqdm import tqdm
from loguru import logger
from argparse import ArgumentParser
from code.db.utils import connect_to_db, COLLECTION_TEXT_MAPPING
from thefuzz import process, fuzz

logger.add(f"{__name__}.log", rotation="500 MB")

UNCREDIBLE_DOMAIN_LABELS = ["conspiracy-pseudoscience", "questionable sources"]


def find_tweets_with_urls(input_collection, output_collection, chunk_size):
    db, client = connect_to_db()
    _input_collection = db[input_collection]
    _output_collection = db[output_collection]

    samples = _input_collection.find({}, batch_size=chunk_size)

    samples_with_urls = []
    for sample in samples:
        url = None

        if "id_str" in sample:
            id_str = sample["id_str"]
        else:
            id_str = sample["id"]

        if "urls" in sample and not type(sample["urls"]) == float:
            url = sample["urls"]

        if not url:
            try:
                url = sample["entities"]["urls"][0]["expanded_url"]

            except (IndexError, KeyError) as e:
                # logger.error(f"There is no url in {id_str}: {e}")
                continue

        samples_with_urls.append({
            "id": sample["id"],
            "id_str": id_str,
            "created_at": sample["created_at"],
            "user_name": sample["user"]["name"] if "user" in sample else sample["user_name"],
            "url": url,
            "full_text": sample[COLLECTION_TEXT_MAPPING[input_collection]]
        })

        if len(samples_with_urls) > chunk_size:
            _output_collection.insert_many(samples_with_urls)
            logger.info(f"{len(samples_with_urls)} samples inserted.")
            samples_with_urls = []

    if len(samples_with_urls) > 0:
        _output_collection.insert_many(samples_with_urls)
        logger.info(f"{len(samples_with_urls)} samples inserted.")

    logger.info(
        f"Samples with urls in {output_collection}: {_output_collection.count_documents({})}")

    client.close()

def find_tweets_without_urls(input_collection, output_collection, chunk_size):
    db, client = connect_to_db()
    _input_collection = db[input_collection]
    _output_collection = db[output_collection]

    samples = _input_collection.find({}, batch_size=chunk_size)

    samples_without_urls = []
    for sample in samples:
        url = None

        # if "id_str" in sample:
        #     id_str = sample["id_str"]
        # else:
        #     id_str = sample["id"]

        if "urls" in sample and not type(sample["urls"]) == float:
            url = sample["urls"]

        if not url or not isinstance(url, dict):
            try:
                url = sample["entities"]["urls"][0]["expanded_url"]

            except:
                pass

        text = sample["text"] if "ebola" in input_collection else sample["full_text"]

        if "http" in text:
            continue

        if not url or isinstance(url, dict):
            del sample["_id"]
            samples_without_urls.append(sample)



        if len(samples_without_urls) > chunk_size:
            _output_collection.insert_many(samples_without_urls)
            logger.info(f"{len(samples_without_urls)} samples inserted.")
            samples_without_urls = []

    if len(samples_without_urls) > 0:
        _output_collection.insert_many(samples_without_urls)
        logger.info(f"{len(samples_without_urls)} samples inserted.")

    logger.info(
        f"Samples without urls in {output_collection}: {_output_collection.count_documents({})}")

    client.close()

def assign_domains(input_collection, chunk_size):
    db, client = connect_to_db()
    _input_collection = db[input_collection]

    count = _input_collection.count_documents({"domain": {"$exists": False}})
    logger.info(f"{count} documents have no domains")

    samples = _input_collection.find({"domain": {"$exists": False}}, batch_size=chunk_size)

    ops = []
    for sample in tqdm(samples, total=count):
        ext = tldextract.extract(sample["url"])
        domain = ext.registered_domain
        ops.append(pymongo.UpdateMany(sample, {"$set": {"domain": domain}}))

        if len(ops) > chunk_size:
            _input_collection.bulk_write(ops)
            logger.info(f"sample {len(ops)} is inserted.")
            ops = []

    if len(ops) > 0:
        _input_collection.bulk_write(ops)
        logger.info(f"sample {len(ops)} is inserted.")

    client.close()


def label_domains(input_collection, chunk_size):
    db, client = connect_to_db()
    _input_collection = db[input_collection]
    mbfc_collection = db['mbfc_labels']

    filter = {"domain": {"$exists": True}}
    domains = _input_collection.distinct("domain", filter)

    all_mbfc_websites = mbfc_collection.distinct("website")

    ops = []

    for domain in tqdm(domains, total=len(domains)):
        candidate_match, score = process.extract(domain, all_mbfc_websites, scorer=fuzz.token_set_ratio, limit=1)[0]

        if score > 99:
            document = mbfc_collection.find_one({"website": candidate_match})

            # If the document exists, retrieve the label
            if document:
                label = document.get("label")
                ops.append(pymongo.UpdateMany({"domain": domain}, {"$set": {"label": label}}))

                if len(ops) > chunk_size:
                    _input_collection.bulk_write(ops)
                    logger.info(f"sample {len(ops)} is inserted.")
                    ops = []

    if len(ops) > 0:
        _input_collection.bulk_write(ops)
        logger.info(f"sample {len(ops)} is inserted.")

    client.close()


def write_specific_fields_to_new_collection(input_collection, chunk_size, output_collection, fields):
    logger.info(f'Selecting {fields}')
    db, client = connect_to_db()
    _input_collection = db[input_collection]
    _output_collection = db[output_collection]

    filter = {"$or": [{"label": field} for field in fields]}

    count = _input_collection.count_documents(filter)

    samples = _input_collection.find(filter, batch_size=chunk_size)

    docs_to_insert = []
    for sample in tqdm(samples, total=count):
        # modify the fields in the document as required
        doc = {key: value for key, value in sample.items() if key != '_id'}
        docs_to_insert.append(doc)

        if len(docs_to_insert) > chunk_size:
            # insert the documents into the output collection
            _output_collection.insert_many(docs_to_insert)
            docs_to_insert = []

    if len(docs_to_insert) > 0:
        _output_collection.insert_many(docs_to_insert)
        logger.info(f"sample {len(docs_to_insert)} is inserted.")

    client.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', help="path to the raw data files")
    parser.add_argument('--input_collection', help="input collection")
    parser.add_argument('--output_collection', help="output collection that contains samples with the urls")
    parser.add_argument('--chunk_size', help="set up chunksize for the large datasets", type=int)
    parser.add_argument('--find_tweets_with_urls', action='store_true')
    parser.add_argument('--find_tweets_without_urls', action='store_true')
    parser.add_argument('--assign_domains', action='store_true')
    parser.add_argument('--label_domains', action='store_true')
    parser.add_argument('--fields', type=lambda s: s.split(","))
    parser.add_argument('--write_specific_fields_to_new_collection', action='store_true')

    args = parser.parse_args()

    if args.find_tweets_with_urls:
        find_tweets_with_urls(input_collection=args.input_collection, output_collection=args.output_collection,
                              chunk_size=args.chunk_size)

    if args.find_tweets_without_urls:
        find_tweets_without_urls(input_collection=args.input_collection, output_collection=args.output_collection,
                              chunk_size=args.chunk_size)

    if args.assign_domains:
        assign_domains(input_collection=args.input_collection, chunk_size=args.chunk_size)

    if args.label_domains:
        label_domains(input_collection=args.input_collection, chunk_size=args.chunk_size)

    if args.write_specific_fields_to_new_collection:
        write_specific_fields_to_new_collection(input_collection=args.input_collection, chunk_size=args.chunk_size,
                                                output_collection=args.output_collection, fields=args.fields)
