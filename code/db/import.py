import pandas as pd
import pymongo
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path

from code.db.utils import MONGO_DB_URL

logger.add(f"{__name__}.log", rotation="500 MB")

JSONL_FILES = ["monkeypox", "zika"]
COVID_FOLDERS = ["hoax_and_hidden_agenda", "vax_misinfo", "media_eval"]

LABELS_MAP = {
    "hoax_and_hidden_agenda": "conspiracy_or_not",
    "vax_misinfo": "is_misinfo"
}


def import_func(args):
    client = pymongo.MongoClient(MONGO_DB_URL)

    assert client.server_info()

    db = client.conspiracy

    data_folder = Path(args.folder)

    if data_folder.is_file():
        collection = db[args.collection]
        read_and_import_chunks(args, collection, data_folder)

    else:

        if args.collection in JSONL_FILES:
            collection = db[args.collection]

            file_aux = "jsonl"

            logger.info(f"Adding {args.collection} files.")

            for path in data_folder.rglob(f"*.{file_aux}"):
                logger.info(f"Retrieving {path}.")
                read_and_import_chunks(args, collection, path)

        else:
            disease_category = data_folder.name

            if disease_category == "covid":

                for covid_folder in COVID_FOLDERS:

                    collection = db[f"{args.collection}_{covid_folder}"]

                    if covid_folder == "hoax_and_hidden_agenda":
                        labels = pd.read_csv(data_folder / covid_folder / "labels.csv")

                        labels = labels[labels["topic"].isin([1, 2])].to_dict('records')

                        labels = {value["tweet_ID"]: "conspiracy" if value["conspiracy_or_not"] == 1 else "non-conspiracy"
                                  for value in labels}

                        # only_include = labels[labels[LABELS_MAP[covid_folder]] == 1]["tweet_ID"].tolist()

                        # only_include = labels["tweet_ID"].tolist()
                        #
                        # read_and_import_chunks(args=args,
                        #                        collection=collection,
                        #                        path=Path(args.folder) / covid_folder / "data.jsonl",
                        #                        filter=only_include)

                        data = pd.read_csv(Path(args.folder) / covid_folder / "data.csv", sep=',')

                        # for chunk_obj in chunk_objs:
                        #     print(chunk_obj.id_str)
                        data['label'] = data.id.apply(lambda x: labels.get(x, 'na'))

                        data = data[data['label'] != 'na']
                        if len(data) > 0:
                            collection.insert_many(data.to_dict('records'))
                            logger.info(f"Inserted {len(data)} files.")

                        # collection.insert_many(data.to_dict('records'))
                        # logger.info(f"Inserted {len(data)} files.")

                    elif covid_folder == "vax_misinfo":
                        labels = pd.read_csv(data_folder / covid_folder / "labels.csv").to_dict('records')

                        labels = {value["id"]: "misinfo" if value["is_misinfo"] == 1 else "non-misinfo" for value in labels}

                        # only_include = labels[labels[LABELS_MAP[covid_folder]] == 1]["id"].tolist()

                        # only_include = labels["id"].tolist()
                        #
                        # read_and_import_chunks(args=args,
                        #                        collection=collection,
                        #                        path=Path(args.folder) / covid_folder / "data.csv",
                        #                        filter=only_include)

                        data = pd.read_csv(data_folder / covid_folder / "data.csv", sep=',')

                        data['label'] = data.id.map(lambda x: labels[x])

                        collection.insert_many(data.to_dict('records'))
                        logger.info(f"Inserted {len(data)} files.")

                    elif covid_folder == "media_eval":

                        def map_label(row):
                            conspiracy = ((row['class_label_for_Suppressed_cures_category'] == 1) &
                                          (row['class_label_for_Behaviour_and_Mind_Control_category'] == 1) &
                                          (row['class_label_for_Antivax_category'] == 1) &
                                          (row['class_label_for_Fake_virus_category'] == 1) &
                                          (row['class_label_for_Intentional_Pandemic_category'] == 1) &
                                          (row['class_label_for_Harmful_Radiation_Influence_category'] == 1) &
                                          (row['class_label_for_Population_reduction_Control_category'] == 1) &
                                          (row['class_label_for_New_World_Order_category'] == 1) &
                                          (row['class_label_for_Satanism_category'] == 1)
                                          )

                            if conspiracy:
                                return 'conspiracy'
                            else:
                                return 'non-conspiracy'

                        data = pd.read_csv(data_folder / covid_folder / "task_1_dev.csv", sep=',')

                        data['label'] = data.apply(lambda row: map_label(row), axis=1)

                        # non_conspiracies = data[
                        #     (data['class_label_for_Suppressed_cures_category'] == 1) &
                        #     (data['class_label_for_Behaviour_and_Mind_Control_category'] == 1) &
                        #     (data['class_label_for_Antivax_category'] == 1) &
                        #     (data['class_label_for_Fake_virus_category'] == 1) &
                        #     (data['class_label_for_Intentional_Pandemic_category'] == 1) &
                        #     (data['class_label_for_Harmful_Radiation_Influence_category'] == 1) &
                        #     (data['class_label_for_Population_reduction_Control_category'] == 1) &
                        #     (data['class_label_for_New_World_Order_category'] == 1) &
                        #     (data['class_label_for_Satanism_category'] == 1)
                        #     ]
                        #
                        # data = data[~data['tweet_id'].isin(non_conspiracies['tweet_id'].tolist())]
                        #
                        # data["label"] = 'no misinfo'

                        collection.insert_many(data.to_dict('records'))
                        logger.info(f"Inserted {len(data)} files.")

            else:
                collection = db[args.collection]

                read_and_import_chunks(args=args,
                                       collection=collection,
                                       path=data_folder / "data.csv")

        logger.info(f"Completed data import for {args.collection}.")

        client.close()

def read_and_import_chunks(args, collection, path, filter=None):
    if ".jsonl" in path.name or "json" in path.name:
        chunk_objs = pd.read_json(path_or_buf=path, lines=True, chunksize=args.chunksize)
    else:
        chunk_objs = pd.read_csv(path, chunksize=args.chunksize)

    for chunk_obj in chunk_objs:
        if filter:
            chunk_obj = chunk_obj[chunk_obj["id_str"].isin(filter)]
            collection.insert_many(chunk_obj.to_dict('records'))
            logger.info(f"Inserted {len(chunk_obj)} files.")
        else:
            chunk_obj.drop(columns=['_id'], inplace=True)
            collection.insert_many(chunk_obj.to_dict('records'))
            logger.info(f"Inserted {len(chunk_obj)} files.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--folder', help="path to the raw data files")
    parser.add_argument('--collection', help="collection name")
    parser.add_argument('--chunksize', help="set up chunksize for the large datasets", type=int)

    args = parser.parse_args()

    import_func(args)
