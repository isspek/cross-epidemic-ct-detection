from abc import abstractmethod
from typing import Tuple

from code.llmag.llmag.common.Identifiable import Identifiable


class LLCompletionModel(Identifiable):
    '''
    Abstract interface for a Large language model (such as GPT-X) that is a text -> text function,
    ie receives input text as input and returns text that is a completion of the input text.
    '''

    @property
    def id(self): return self._id

    def __init__(self, id):
        self._id = id

    @abstractmethod
    def __call__(self, txt: str) -> Tuple[str, object]:
        '''
        For a text return text output and an object representing full response of the concrete model instance's.
        '''