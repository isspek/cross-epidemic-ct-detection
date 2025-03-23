from code.db.utils import SENTENCE_EMBEDDING_DIM
import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import os
import pandas as pd
import random
import torch
from argparse import ArgumentParser
from docarray import Document, DocumentArray
from loguru import logger
from sentence_transformers import util
from transformers import pipeline
from tqdm import tqdm
from summarizer import Summarizer

logger.add(f"{__name__}.log", rotation="500 MB")


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def cluster(search_collection, output_file, cuda_device):
    # da = DocumentArray(
    #     storage='elasticsearch',
    #     config={
    #         'hosts': ES_HOST,
    #         'n_dim': int(SENTENCE_EMBEDDING_DIM),
    #         'index_name': search_collection,
    #         'es_config': {'request_timeout': 120},
    #         'index_text': True
    #     },
    # )

    da = DocumentArray(
        storage='redis',
        config={'host': 'localhost', 'port': 6379, 'n_dim': int(SENTENCE_EMBEDDING_DIM),
                'index_name': search_collection, 'index_text': True},

    )

    embeddings = da.embeddings

    logger.info(f'Number of documents for clustering: {len(embeddings)}')

    ## Leiden community detection
    logger.info('Started leiden community detection.')

    cos_mat = util.cos_sim(da.embeddings, da.embeddings)
    cos_copy = cos_mat.detach().cpu().numpy()
    cos_copy[cos_copy < 0.85] = 0

    G = nx.from_numpy_array(cos_copy)
    g = ig.Graph.from_networkx(G)
    communities = la.find_partition(g, la.ModularityVertexPartition, n_iterations=-1, seed=args.random_seed)

    optimizer = la.Optimiser()
    optimizer.set_rng_seed(args.random_seed)

    diff = 1
    while diff > 0:
        diff = optimizer.optimise_partition(communities, n_iterations=-1)

    community_labeled_data = []
    k = 0
    for i in range(len(communities)):
        if (len(communities[i]) > 4):
            docs = da[communities[i]]
            for doc in docs:
                community_labeled_data.append(
                    {
                        "cleaned_text": doc.text,
                        "community_label": i
                    }
                )
            k += 1

    df = pd.DataFrame(community_labeled_data)



    logger.info(f'From {len(da.embeddings)} samples, {len(df.community_label.unique())} communities are detected.')


    summarizer = pipeline('summarization', model="facebook/bart-large-cnn", device=int(cuda_device))

    model = Summarizer(model='distilbert-base-uncased')

    new_data = []
    for name, group in tqdm(df.groupby('community_label'), total=len(df.community_label.unique())):
        texts_len = np.asarray([len(text) for text in group.cleaned_text.to_list()])

        max_length = int(np.mean(texts_len))
        min_length = int(np.min(texts_len))
        texts = [f'{sample}.' for sample in group.cleaned_text.to_list()]
        text = ' '.join(sample for sample in texts)

        summary = summarizer(text, truncation=True, min_length=min_length, max_length=len(text) if len(text) < max_length else max_length)
        group['abs-summary'] = summary[0]['summary_text']
        group['ext-summary'] = model(text, num_sentences=1)
        new_data.append(group)

    pd.concat(new_data).to_csv(output_file, sep='\t', index=False)

    logger.info('Saved the community labels and their summarization.')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda_device', type=str)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--chunk_size', type=int)
    parser.add_argument('--collection', type=str)
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--search_collection', type=str)
    parser.add_argument('--embedding_name', type=str)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    set_random_seed(args.random_seed)

    cluster(search_collection=args.search_collection, output_file=args.output_file, cuda_device=args.cuda_device)
