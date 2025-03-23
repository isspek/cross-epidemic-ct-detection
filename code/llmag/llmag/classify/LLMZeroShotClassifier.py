from code.llmag.llmag.common.Identifiable import Identifiable
from code.llmag.llmag.corpus.Corpus import Corpus
from code.llmag.llmag.llm import LLCompletionModel
from code.llmag.llmag.prompt.LLMZeroShotTemplate import LLMZeroShotTemplate


class LLMZeroShotClassifer(Identifiable):
    ''' Classify texts based on a LLM, and a prompt-building template. '''

    @property
    def id(self):
        if not hasattr(self, '_id'): self._id = self._compose_id()
        return self._id

    def _compose_id(self):
        return f'{self._llm.id}-{self._prompt_tmpl.id}-{self._corpus.id}'

    def __init__(self, llm: LLCompletionModel, prompt_tmpl: LLMZeroShotTemplate, corpus: Corpus):
        '''
        :param llm: LLM to be used for prompting
        :param prompt_tmpl: template for creating LLM prompts
        :param corpus: Corpus that enables fetching texts by id
        '''
        self._llm = llm
        self._prompt_tmpl = prompt_tmpl
        self._corpus = corpus

    def __call__(self, txt_id):
        '''
        :param txt_id: id of a text from the corpus
        :return:
        '''
        txt = self._corpus[txt_id]
        prompt = self._prompt_tmpl.create_prompt(txt)
        raw_answ, response = self._llm(prompt)
        cls_answ = self._prompt_tmpl.class_from_output(raw_answ)
        return cls_answ, raw_answ, response, prompt