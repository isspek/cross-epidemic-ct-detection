import pandas as pd

from code.corpora.data_utils import dframe_info
from code.pysettings import TOY_DATASET
# from code.pysettings import *


def load_excel_dset(path, id_col='id_str', sep='\t'):
    return pd.read_csv(path, sep=sep, engine='python', dtype={id_col: str})

def toy_dset():
    df = load_excel_dset(TOY_DATASET)
    return df

def toy_dset_list():
    return list(toy_dset()['cleaned_text'])

if __name__ == '__main__':
    dframe_info(load_excel_dset(TOY_DATASET))