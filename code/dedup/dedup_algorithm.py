import numpy as np, pandas as pd, seaborn as sns
#import cupy as cp
import matplotlib.pyplot as plt
from pathlib import Path

import pandas
from scipy.spatial.distance import cdist as npcdist, cosine
#from cupyx.scipy.spatial.distance import cdist as cpcdist
from sklearn.feature_extraction.text import CountVectorizer
import igraph as ig

from code.corpora.data_loaders import toy_dset, load_excel_dset
from code.corpora.data_utils import set_random_seed
from code.dedup.levenshtein import LevenshteinDistance

import _pickle

from code.dedup.db_reader import iterate_samples
# from code.pysettings import *


def collection_to_dataframe(name, txt_col='cleaned_text', id_col='id_str'):
    '''
    Convert (mongodb) collection of texts to a dataframe with specified columns for id and text.
    '''
    ids, txts = [], []
    for id, txt in iterate_samples(input_collection=name):
        ids.append(id); txts.append(txt)
    return pandas.DataFrame({ txt_col: txts, id_col: ids })

def build_id_graph(df, out_folder='.', output_stats=False, verbose=False):
    ids = set(df['id1']).union(set(df['id2']))
    ids = list(ids)
    g = ig.Graph()
    g.add_vertices(len(ids))
    g.vs['name'] = ids
    if verbose:
        print('Unconnected vertices: ')
        for vx in g.vs: print(vx.index, vx.attributes()['name'])
    id2ix = { id:ix for ix, id in enumerate(ids) }
    edges = [(id2ix[r['id1']], id2ix[r['id2']]) for _, r in df.iterrows()]
    g.add_edges(edges)
    if verbose:
        for _, r in df.iterrows():
            print(f'edge: {id2ix[r["id1"]]}-{id2ix[r["id2"]]}; {r["id1"]}-{r["id2"]}')
    if output_stats: # output conn. components stats
        fname1 = Path(out_folder)/'conn_comp_len_distrib.txt'
        components = g.connected_components(mode='strong')
        cluster_lens = [len(c) for c in components]
        with open(fname1, 'w') as f:
            print(pd.Series(cluster_lens, dtype=int).describe(), file=f)
        fname2 = Path(out_folder) / 'conn_comp_len_distrib.png'
        sns.histplot(cluster_lens, discrete=True)
        #plt.xticks(range(1, max(cluster_lens)+2))
        plt.tight_layout()
        plt.savefig(fname2, dpi=300)
    return g

def create_load_BOW(texts: pd.Series, out_folder: Path = None, lowercase = True):
    if out_folder: # load saved vectors, if they exist
        outf = out_folder/'bow.pickle'
        if outf.exists(): return _pickle.load(open(outf, 'rb'))
    if lowercase: texts = texts.apply(lambda x: x.lower())
    print('Starting dedup, num. texts: ', len(texts))
    txt2vec = CountVectorizer(lowercase=False)
    print('Vectorizing texts ...')
    txt_vecs = txt2vec.fit_transform(texts).toarray()
    if out_folder: _pickle.dump(txt_vecs, open(outf, 'wb'))
    return txt_vecs

def create_load_cosd_pairs(vecs, dist_thresh=0.1, engine='np', out_folder: Path = None):
    '''
    Calculate pairwise cosine distances and select pairs by distance threshold.
    :param vecs: matrix, vectors are rows
    :param engine: 'cp' (cupy) or 'np' (numpy)
    :return: list of [(i1, i2)] - indices of all pairs with cosine dist. below treshold
    '''
    if out_folder: # load saved pairs, if they exist
        out_file = out_folder/f'cos_pairs_{dist_thresh:.2f}.pickle'
        if out_file.exists(): return _pickle.load(open(out_file, 'rb'))
    print('Calculating cosine distances ...')
    if engine == 'cp':
        raise ValueError('cp engine not implemented')
        # cp testing with dot product
        # vecs = cp.array(vecs)
        # print(vecs.shape)
        # print(vecs.transpose().shape)
        # cos_dists = cp.dot(vecs, vecs.transpose())
        # TODO: write a function that accepts two (row) matrices X, Y, algo is:
        # calc norms
        # np.sqrt((X * X).sum(axis=1))
        # or:
        # norms = np.einsum("ij,ij->i", X, X)
        # np.sqrt(norms, norms)
        # X /= norms[:, np.newaxis]
        #
        # algo:
        # normalize X, Y
        # calculate S = X dot Y (cosine sim.)
        # S *= -1 ? why not one step
        # S += 1
        # np.clip(S, 0, 2, out=S)
        # S[np.diag_indices_from(S)] = 0.0
        # print(pd.Series(cos_dists.get().flatten()).describe())
        rows, cols = cp.where(cos_dists < dist_thresh)
        rows, cols = rows.get(), cols.get() # transfer from GPU
    else:
        cos_dists = npcdist(vecs, vecs, metric='cosine')
        rows, cols = np.where(cos_dists < dist_thresh)
    res = np.array([(r, c) for r, c in zip(rows, cols) if r < c ])
    if out_folder: _pickle.dump(res, open(out_file, 'wb'))
    return res

def create_load_filtered_pairs(texts: pd.Series, pairs, txt_vecs, df: pd.DataFrame, id_col: str, out_folder: Path):
    if out_folder: # load saved vectors, if they exist
        outf = out_folder/'pairdata.xlsx'
        if outf.exists(): return pd.read_excel(outf, dtype={'id1':str, 'id2':str})
    columns = ['cos_dist', 'lev_dist_wrd', 'lev_dist_chr',
               'lev_wrd_as_perc_of_shorter', 'lev_chr_as_perc_of_shorter',
               'id1', 'id2', 'tokens1', 'tokens2', 'chars1', 'chars2']
    results_df = pd.DataFrame(columns=columns)
    print('Formatting data for the analysis...')
    # Calculate lev. distances and filter pairs
    wstokenizer = lambda txt: txt.split()
    levDistWord = LevenshteinDistance(txt2words=wstokenizer)
    levDistChar = LevenshteinDistance(txt2words=lambda txt: [ch for ch in txt])
    for r, c in pairs:
        if (r < c):
            ix1, ix2 = texts.index[r], texts.index[c]
            assert(texts[ix1] == texts.iloc[r])
            assert(texts[ix2] == texts.iloc[c])
            txt1, txt2 = texts[ix1], texts[ix2]
            id1, id2 = df[id_col][ix1], df[id_col][ix2]
            lev_dist_wrd = levDistWord(txt1, txt2)
            lev_dist_chr = levDistChar(txt1, txt2)
            tok1, tok2 = len(wstokenizer(txt1)), len(wstokenizer(txt2))
            chars1, chars2 = len(txt1), len(txt2)
            lev_wrd_perc = lev_dist_wrd/min(tok1, tok2)*100
            lev_chr_perc = lev_dist_chr/min(chars1, chars2)*100
            if lev_wrd_perc <= 5 or lev_chr_perc <= 5:
                row = pd.DataFrame({
                    'id1': [str(id1)], 'id2': [str(id2)],
                    'tokens1': [tok1], 'tokens2': [tok2], 'chars1': [chars1], 'chars2': [chars2],
                    'cos_dist': [cosine(txt_vecs[r], txt_vecs[c])],
                    'lev_dist_wrd': [lev_dist_wrd],
                    'lev_dist_chr': [lev_dist_chr],
                    'lev_wrd_as_perc_of_shorter': [lev_wrd_perc],
                    'lev_chr_as_perc_of_shorter': [lev_chr_perc],
                })
                results_df = pd.concat([results_df, row], ignore_index=True)
    if out_folder:
        results_df.to_excel(outf, index=False)
    return results_df

def plot_ccs(ccs, it, out_folder):
    fig, ax = plt.subplots()
    ig.plot(
        ccs,
        target=ax,
        palette=ig.RainbowPalette(),
        vertex_size=0.07,
        vertex_color=list(map(int, ig.rescale(ccs.membership, (0, 200), clamp=True))),
        edge_width=0.7
    )
    plt.tight_layout()
    if out_folder is None: out_folder = Path('.')
    fname = out_folder / f'conn_components_{it}.png'
    plt.savefig(fname, dpi=300)

def dedup_on_graph(g: ig.Graph, verbose=False, out_folder=None):
    #     // if component is of size 2, chose the representative at random
    #     update the connected components - remove old, add subcomponent
    rep_nodes, duplicate_ids = [], {}
    # 5. Do until there are only isolated vertices (no edges)
    it = 0
    out_file = Path(out_folder) / 'result.pickle'
    if out_folder: # load saved result
        if out_file.exists(): return _pickle.load(open(out_file, 'rb'))
    while True:
        if verbose: print('\nNew Cycle.....')
        it += 1
        ccs = g.connected_components(mode='strong')
        if verbose: plot_ccs(ccs, it, out_folder)
        if (len(ccs) == 0): break
        #  find a connected component > 1
        if verbose: print(f'Num CC: {len(ccs)}')
        cc_found = False
        for cc in ccs:
            ccg = g.induced_subgraph(cc)
            cc_size = ccg.vcount()
            if cc_size > 1:
                cc_found = True
                if verbose: print(f'CC SIZE: {cc_size}')
                # find text/node with the largest degree
                max_v, max_deg = None, -1
                for v in ccg.vs:
                    if v.degree() > max_deg: max_deg, max_v = v.degree(), v
                # add max.deg node as a representative text
                if verbose: print(f'Max deg V: {max_v}, degree: {max_deg}')
                rep_id = max_v.attributes()['name']
                rep_nodes.append(rep_id)
                neighs = max_v.neighbors()
                duplicate_ids[rep_id] = [n.attributes()['name'] for n in neighs]
                if verbose: print('Neighbours: ', duplicate_ids[rep_id])
                # remove the node and its neighbours (duplicates) from the graph
                rem = [rep_id]
                rem.extend(duplicate_ids[rep_id])
                rem_ixs = [g.vs.find(name=id).index for id in rem]
                # graph is reset, go to outer loop (to recalc. CCs)
                g.delete_vertices(rem_ixs)
                break
        if not cc_found: break
    if out_folder: _pickle.dump((rep_nodes, duplicate_ids), open(out_file, 'wb'))
    return rep_nodes, duplicate_ids


def create_deduplication_report(df: pd.DataFrame, rep_ids, dup_ids, id_col, txt_col, out_folder):
    '''
    Create a report given the ids of the texts that *participated in the deduplication process*,
    ie, the texts that were paired with another close text (in create_load_filtered_pairs).
    The ids in the dup_ids data structure are the one that need to be removed from the entire corpus.
    :param df: dataframe with texts, containing the id_col columnt with the IDs
    :param rep_ids: IDs of the representative (kept) texts
    :param dup_ids: map of RID -> list of IDs of RID's duplicates
    :return:
    '''
    df = df.set_index(id_col, verify_integrity=True)
    rep_ids = sorted(rep_ids, key=lambda i: df.loc[i, txt_col])
    REP = len(set(rep_ids))
    assert REP == len(rep_ids)
    dup_list = [did for did_lst in dup_ids.values() for did in did_lst]
    DUP = len(set(dup_list))
    assert DUP == len(dup_list)
    outf = Path(out_folder) / 'deduplication_report.txt'
    with open(outf, 'w') as f:
        print(f'Kept {REP} representative texts, removed {DUP} duplicates,  texts\' strings are delimited with []\n', file=f)
        for rid in rep_ids:
            print(f'repr.txt [{df.loc[rid, txt_col]}]', file=f)
            for did in dup_ids[rid]:
                print(f'       d [{df.loc[did, txt_col]}]', file=f)
            print(file=f)


def load_static_dedup_dframe(name, store_folder):
    file = Path(store_folder)/f'{name}.csv'
    df = load_excel_dset(file, sep=',')
    return df

def run_deduplication_on_collection(name, store_folder='.', txt_col='cleaned_text', id_col='id_str', engine='np',
                                    cos_filter_thresh=0.1, lowercase=True, subsample=None, remove_exact=True,
                                    rnd_seed=88103, verbose=False, report=True):
    '''
    Run full deduplication algorithm on a collection.
    Store intermediate results (for quick re-run) and final results in the output_folder's collection-subfolder.
    :param name: name of the collection
    :param store_folder:
    :return:
    '''
    # load data
    if name == 'toy_dset': df = toy_dset()
    else:
        #df = collection_to_dataframe(name, txt_col=txt_col, id_col=id_col)
        df = load_static_dedup_dframe(name, store_folder)
    set_random_seed(rnd_seed)
    if subsample:
        df = df.sample(subsample)
        df.reset_index(inplace=True)
    if remove_exact:
        print('Removing exact duplicates ...')
        len_bf = len(df)
        df = remove_exact_duplicates(df, txt_col=txt_col, lowercase=lowercase)
        print(f'... done, old length {len_bf}, new length {len(df)}')
    # create folder for storing output files
    out_folder  = Path(store_folder) / name
    out_folder.mkdir(exist_ok=True)
    # 1. Vectorize texts (bag-of-words)
    texts = df[txt_col]
    txt_vecs = create_load_BOW(texts, out_folder=out_folder, lowercase=lowercase)
    # 2. Calculate all pairwise cosine distances, take those with distance < 0.1
    pairs = create_load_cosd_pairs(txt_vecs, dist_thresh=cos_filter_thresh, engine=engine, out_folder=out_folder)
    # 3. For the similar pairs, filter pairs based on levenshtein distances
    filt_pairs = create_load_filtered_pairs(texts=texts, pairs=pairs, txt_vecs=txt_vecs, df=df, id_col=id_col,
                                            out_folder=out_folder)
    # 4. Create a graph where "same" texts are connected, calculate connected components
    graph = build_id_graph(filt_pairs, output_stats=True, out_folder=out_folder, verbose=verbose)
    # 5. graph-based deduplication
    rep_ids, dup_ids = dedup_on_graph(graph, out_folder=out_folder, verbose=verbose)
    if report:
        create_deduplication_report(df, rep_ids, dup_ids, id_col=id_col, txt_col=txt_col, out_folder=out_folder)
    return rep_ids, dup_ids

def remove_exact_duplicates(df, txt_col='cleaned_text', lowercase=True):
    '''
    Deduplicate by exact match of txt_col.
    :return: dataframe with only one example per duplicate
    '''
    buckets = {}
    for ix, row in df.iterrows():
        txt = row[txt_col].lower() if lowercase else row[txt_col]
        h = hash(txt)
        bucket = buckets[h] if h in buckets else []
        bucket.append((txt, ix))
        buckets[h] = bucket
    dedup_ixs = []
    for buck in buckets.values():
        seen = set()
        for txt, tid in buck:
            if txt not in seen:
                seen.add(txt)
                dedup_ixs.append(tid)
    assert len(dedup_ixs) == len(set(dedup_ixs)) # sanity check
    result = df.iloc[dedup_ixs].copy()
    result.reset_index(inplace=True)
    return result

def test_dedup():
    run_deduplication_on_collection('zika_with_urls_candidates', subsample=3000, engine='np')
    #analyze_deduplication('toy_dset_dedup_analysis.xlsx')

def load_result(collection='ebola', store_folder='.'):
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

if __name__ == '__main__':
    # run full dedup on a  folder:
    # dataset_files = ['ebola_with_urls_candidates','zika_with_urls_candidates', 'monkeypox_with_urls_candidates']
    dataset_files = ['monkeypox_with_urls_candidates']
    for file_name in dataset_files:
        run_deduplication_on_collection(file_name, store_folder='data/candidates',subsample=None, engine='np', verbose=False)
        # just load the results:
        # load_result(file_name)