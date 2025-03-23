import pandas as pd, seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
import igraph as ig

from code.corpora.data_utils import set_random_seed
from code.dedup.dedup_algorithm import remove_exact_duplicates, create_load_cosd_pairs
from code.dedup.levenshtein import LevenshteinDistance

from code.corpora.data_loaders import toy_dset


def create_dedup_data(df, out_fname, txt_col='cleaned_text', id_col='id_str', engine='np',
                      cos_filter_thresh=0.1, lowercase=True, subsample=None, remove_exact=True, rnd_seed=88103):
    '''
    Perform tentative deduplication via cosine distance and levenshtein distance.
    Collect possible duplicate pairs and store them in a table together with other stats.
    '''
    set_random_seed(rnd_seed)
    if subsample:
        df = df.sample(subsample)
        df.reset_index(inplace=True)
    if remove_exact:
        print('Removing exact duplicates ...')
        len_bf = len(df)
        df = remove_exact_duplicates(df, txt_col)
        print(f'... done, old length {len_bf}, new length {len(df)}')
    texts = df[txt_col]
    if lowercase: texts = texts.apply(lambda x: x.lower())
    print('Starting dedup, num. texts: ', len(df))
    wstokenizer = lambda txt: txt.split()
    charsplitter = lambda txt: [ch for ch in txt]
    levDistWord = LevenshteinDistance(txt2words=wstokenizer)
    levDistChar = LevenshteinDistance(txt2words=charsplitter)
    txt2vec = CountVectorizer(lowercase=False)
    print('Vectorizing texts ...')
    txt_vecs = txt2vec.fit_transform(texts).toarray()
    pairs = create_load_cosd_pairs(txt_vecs, dist_thresh=cos_filter_thresh, engine=engine)
    columns = ['cos_dist', 'lev_dist_wrd', 'lev_dist_chr',
               'lev_wrd_as_perc_of_shorter', 'lev_chr_as_perc_of_shorter',
               'id1', 'txt1', 'id2', 'txt2', 'tokens1', 'tokens2']
    results_df = pd.DataFrame(columns=columns)
    print('Formatting data for the analysis...')
    for r, c in pairs:
        #c = cols[i]
        if (r < c):
            ix1, ix2 = texts.index[r], texts.index[c]
            print(r, ix1, c, ix2)
            assert(texts[ix1] == texts.iloc[r])
            assert(texts[ix2] == texts.iloc[c])
            txt1, txt2 = texts[ix1], texts[ix2]
            id1, id2 = df[id_col][ix1], df[id_col][ix2]
            lev_dist_wrd = levDistWord(txt1, txt2)
            lev_dist_chr = levDistChar(txt1, txt2)
            tok1, tok2 = len(wstokenizer(txt1)), len(wstokenizer(txt2))
            lev_wrd_perc = lev_dist_wrd/min(tok1, tok2)*100
            lev_chr_perc = lev_dist_chr/min(len(txt1), len(txt2))*100
            row = pd.DataFrame({
                'cos_dist': [cosine(txt_vecs[r], txt_vecs[c])],
                'lev_dist_wrd': [lev_dist_wrd],
                'lev_dist_chr': [lev_dist_chr],
                'lev_wrd_as_perc_of_shorter': [lev_wrd_perc],
                'lev_chr_as_perc_of_shorter': [lev_chr_perc],
                'id1': [id1], 'txt1': [txt1], 'id2': [id2], 'txt2': [txt2],
                'tokens1': [tok1], 'tokens2': [tok2]
            })
            results_df = pd.concat([results_df, row], ignore_index=True)
    out_fname = Path(out_fname).name
    results_df.to_excel(out_fname, index=False)


def graph_analysis(df, fname):
    ids = set(df['id1']).union(set(df['id2']))
    g = ig.Graph()
    g.add_vertices(list(ids))
    g.add_edges([(r['id1'], r['id2']) for _, r in df.iterrows()])
    components = g.connected_components(mode='strong')
    cluster_lens = [len(c) for c in components]
    with open(f'{fname}_conn_comp_len_dist.txt', 'w') as f:
        print(pd.Series(cluster_lens).describe(), file=f)
    #fig, ax = plt.subplots()
    sns.histplot(cluster_lens)
    #plt.xticks(range(1, max(cluster_lens)+2))
    plt.tight_layout()
    plt.savefig(f'{fname}.conn_comp_len_dist.png', dpi=300)

def analyze_deduplication(fname):
    '''
    Print statistics and output from the dedup data table created with create_dedup_data().
    :param fname: name of the stored xlsx table
    :return:
    '''
    df = pd.read_excel(fname, dtype={'id1':str, 'id2':str})
    graph_analysis(df, fname)
    fname = Path(fname).name
    def print_dedup_row(ix, r, f):
        print(f'pair no. {ix:5}; cosine dist: {r["cos_dist"]:.3f}, lev.word: {r["lev_dist_wrd"]}, lev.char: {r["lev_dist_chr"]}, '
              f'lev.wrd % of shorter: {r["lev_wrd_as_perc_of_shorter"]:.3f}%, '
              f'lev.chr % of shorter: {r["lev_chr_as_perc_of_shorter"]:.3f}%', file=f)
        print(f'id: {r["id1"]},', r["txt1"], file=f)
        print(f'id: {r["id2"]},', r["txt2"], file=f)
        print(file=f)
    # print pairs sorted by cosine and levenshtein distance
    for col in ['cos_dist', 'lev_dist_wrd', 'lev_dist_chr']:
        with open(f'{fname}.sort_by.{col}.txt', 'w') as f:
            for i, (ix, r) in enumerate(df.sort_values(col).iterrows()):
                print_dedup_row(i+1, r, f)
    # blot distribution of the distances
    for col in ['cos_dist', 'lev_dist_wrd']:
        fig, ax = plt.subplots(1, 2)
        sns.boxplot(df, y=col, ax=ax[0])
        sns.histplot(df, x=col, ax=ax[1])
        plt.tight_layout()
        plt.savefig(f'{fname}.{col}.dist.png', dpi=300)
    fig, ax = plt.subplots()
    sns.regplot(df, x='lev_dist_wrd', y='cos_dist')
    plt.tight_layout()
    plt.savefig(f'{fname}.lev_wrd_X_cos_dist_corr.png', dpi=300)

def test_dedup():
    create_dedup_data(toy_dset(), subsample=1000, engine='cp', cos_filter_thresh=0.05,
                      out_fname='toy_dset_dedup_analysis.xlsx', )
    analyze_deduplication('toy_dset_dedup_analysis.xlsx')

if __name__ == '__main__':
    test_dedup()
    #create_dedup_data(toy_dset(), subsample=None, out_fname='toy_dset_dedup_analysis.xlsx')
    #analyze_deduplication('toy_dset_dedup_analysis.xlsx')