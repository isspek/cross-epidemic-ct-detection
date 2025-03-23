import pandas as pd

from code.pysettings import CLUSTERING_TSV_V1

def load_clustering_df(path):
    return pd.read_csv(path, sep='\t', engine='python')

def print_grouped_clusters(path):
    df = load_clustering_df(path)
    clusters = {}
    for _, r in df.iterrows():
        cid, txt = int(r['community_label']), r['cleaned_text']
        clusters[cid] = clusters.get(cid, [])
        clusters[cid].append(txt)
    for cid in sorted(clusters.keys()):
        print(f'CLUSTER {cid}')
        for txt in sorted(clusters[cid]):
            print('  ', txt)
        print()

if __name__ == '__main__':
    print_grouped_clusters(CLUSTERING_TSV_V1)