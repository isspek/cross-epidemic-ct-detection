from typing import List, Tuple

from llmag.common.Identifiable import Identifiable
from llmag.corpus.Corpus import Corpus
from llmag.llm import LLCompletionModel
from llmag.prompt.LLMFewShotTemplate import LLMFewShotTemplate
from llmag.llmag_utils.misc_utils import str_hash


class LLMFewShotClassifer(Identifiable):
    ''' Classify texts based on a LLM, and a prompt-building template. '''

    @property
    def id(self):
        if not hasattr(self, '_id'): self._id = self._compose_id()
        return self._id

    def _compose_id(self):
        exmpl_id = str_hash(','.join([f'{ex[0]}:{ex[1]}' for ex in self._examples]))
        return f'{self._llm.id}-{self._prompt_tmpl.id}-{self._corpus.id}-{exmpl_id}'

    def __init__(self, llm: LLCompletionModel, prompt_tmpl: LLMFewShotTemplate, corpus: Corpus, examples: List[Tuple[object, str]]):
        '''
        :param llm: LLM to be used for prompting
        :param prompt_tmpl: template for creating LLM prompts
        :param corpus: Corpus that enables fetching texts by id
        :param examples: list of few-shot examples, given as list of (text_id, class_label), has to be in line with prompt labels
        '''
        self._llm = llm
        self._prompt_tmpl = prompt_tmpl
        self._corpus = corpus
        self._examples = examples

    def __call__(self, txt_id):
        '''
        :param txt_id: id of a text from the corpus
        :return:
        '''
        txt = self._corpus[txt_id]
        # generate examples by fetching texts from corpus
        examples = [(self._corpus[ex[0]], ex[1]) for ex in self._examples]
        prompt = self._prompt_tmpl.create_prompt(examples, txt)
        raw_answ, response = self._llm(prompt)
        cls_answ = self._prompt_tmpl.class_from_output(raw_answ)
        return cls_answ, raw_answ, response, prompt