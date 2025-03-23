from abc import abstractmethod

from code.llmag.llmag.common.Identifiable import Identifiable


class Corpus(Identifiable):
    '''
    Defines interface of Corpus-like objects.
    Corpus represents a collection of Text-like objects and
     provides methods for fetching texts and iterating over texts.
    Each text in a corpus is expected to have a unique id.
    '''

    @property
    @abstractmethod
    def id(self): ...

    @abstractmethod
    def get_text(self, id):
        '''Fetch single text by id. '''

    def __getitem__(self, id):
        ''' Alias of get_text, for syntactic sugar '''
        return self.get_text(id)

    def get_texts(self, ids):
        '''Fetch texts by iterable of ids.
        Default implementation uses get_text(). '''
        return [self.get_text(id) for id in ids]

    @abstractmethod
    def __iter__(self):
        '''Iterate over all the texts in the corpus. Each text must be provided only once.'''

    def text_ids(self):
        '''get ids of all the texts in the corpus'''
        return [ txto.id for txto in self ]

    def __len__(self):
        if not hasattr(self, '_length'):
            self._length = sum(1 for _ in self)
        return self._length

    def __contains__(self, txt_id):
        ''' Check if item is a text id in the corpus '''
        return txt_id in self.text_ids()

if __name__ == '__main__':
    pass
