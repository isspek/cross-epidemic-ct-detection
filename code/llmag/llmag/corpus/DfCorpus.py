import pandas as pd

from llmag.corpus.Corpus import Corpus
from llmag.corpus.Text import SimpleText


class DfCorpus(Corpus):
    '''
    Implementation of a corpus with texts (and related data) residing in a Pandas DataFrame
    '''

    def __init__(self, cid, df : pd.DataFrame, id_col, txt_col, properties=None):
        '''
        :param cid: Id of the corpus
        :param df: dataframe with texts
        :param id_col: column of df containing *unique* text ids
        :param txt_col: column of df containing texts
        :param properties: if not None, then text's attributes in the dataframe are added as Text objects' properties.
            can be True (add columns != id_col, txt_col), or a list of ids
        '''
        self._id = cid
        self._df = df
        self._idc, self._txtc = id_col, txt_col
        self._df.set_index(id_col, verify_integrity=True) # verify id uniqueness
        self._process_props(properties)

    def _process_props(self, props):
        if not props:
            self._props = None
        else:
            cols = [c for c in self._df if c not in [self._idc, self._txtc]]
            if props == True: self._props = cols # add all
            else:
                self._props = []
                for prop in props:
                    if prop in cols: self._props.append(prop)
                    else: raise ValueError(f'Property {prop} is not a column of a dataframe.')

    @property
    def id(self): return self._id

    def _get_text_props(self, id):
        if self._props is None: return {}
        else:
            return { p: self._df.loc[id, p] for p in self._props }

    def get_text(self, id):
        '''Fetch single text by id. '''
        if id in self._df.index:
            strtxt = self._df.loc[id, self._txtc]
            props = self._get_text_props(id)
            return SimpleText(id, strtxt, **props)
        else: return None

    def get_texts(self, ids):
        '''Fetch texts by iterable of ids.
        Default implementation uses get_text(). '''
        return [self.get_text(id) for id in ids]

    def __iter__(self):
        '''Iterate over all the texts in the corpus. Each text must be provided only once.'''
        for id in self._df.index:
            yield self.get_text(id)

    def text_ids(self):
        '''get ids of all the texts in the corpus'''
        return list(self._df.index)

    def __contains__(self, txt_id):
        return txt_id in self._df.index

    def __len__(self):
       return len(self._df)
