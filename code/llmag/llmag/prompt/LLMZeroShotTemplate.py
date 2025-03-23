import warnings
from typing import List, Union, Tuple

from llmag.llmag_utils.misc_utils import label_normalize
from llmag.prompt.LLMPromptTemplate import LLMPromptTemplate


class LLMZeroShotTemplate(LLMPromptTemplate):
    '''
    Template for LLM zero-shot classification, creates text prompt form problem description and class labels,
     and maps the LLM's output to labels.
    '''

    def __init__(self, task_id, cls_defs: Union[List[str], List[Tuple[str, str]]], task_desc=None,
                 txt_desc='text', class_desc='class', delimiter=None, default_cls=None):
        '''
        :param task_id: id of the classification problem
        :param task_desc: description of the classification problem
        :param cls_defs: defines class labels, which should be short understandable phrases
                if descriptions are given, they will be used for the default problem description
        :param txt_desc: label to go before texts, ex. TWEET
        :param class_desc: short descr. to go before class label, ex: SENTIMENT
        :param delimiter: string to delimit the final query example from the rest
        :param default_cls: class to return if output cannot be matched to any class
        '''
        super().__init__(task_id=task_id, task_desc=task_desc)
        self._txt_desc = txt_desc; self._idable('_txt_desc')
        self._cls_desc = class_desc; self._idable('_cls_desc')
        self._delimiter = delimiter; self._idable('_delimiter')
        self._check_and_assign_class_vars(cls_defs, default_cls)
        self._task_desc = task_desc if task_desc else self._default_problem_description()
        self._idable('_task_desc')
        self._idable('DEFAULT_TASK_DESC_TEMPL')

    def _check_and_assign_class_vars(self, cls_defs, default_cls):
        # assign
        if len(cls_defs) < 2: raise ValueError('there must be at least two classes')
        if isinstance(cls_defs[0], tuple):
            self._classes = [cd[0] for cd in cls_defs]
            if len(self._classes) != len(set(self._classes)): raise ValueError('class labels must be unique, not repeated')
            self._cls_def = {cd[0]: cd[1] for cd in cls_defs}
        else:
            self._classes = cls_defs
            self._cls_def = {c: None for c in self._classes}
        self._idable('_classes'); self._idable('_cls_def')
        self.C = len(self._classes)
        # check
        if default_cls not in self._classes: raise ValueError('default class must be among classes')
        self._default_cls = default_cls; self._idable('_default_cls')
        if self._task_desc: # custom task description, do some checking
            for c in self._classes:
                if c not in self._task_desc: warnings.warn(f'class label "{c}" not found in task description')
            if self._txt_desc not in self._task_desc:
                warnings.warn(f'text description "{self._txt_desc}" not found in task description')
            if self._cls_desc not in self._task_desc:
                warnings.warn(f'class description "{self._cls_desc}" not found in task description')

    def _str_cls_desc(self, cls):
        return f'{cls}' + (f' ({self._cls_def[cls]})' if self._cls_def[cls] else '')

    DEFAULT_TASK_DESC_TEMPL = 'Determine whether the {txt_desc} is {cls_desc}'

    def _default_problem_description(self):
        cd = ', '.join(self._str_cls_desc(c) for c in self._classes[:-1])
        cd += f' or {self._str_cls_desc(self._classes[-1])}'
        dd = self.DEFAULT_TASK_DESC_TEMPL.format(txt_desc=self._txt_desc, cls_desc=cd)
        return dd

    def create_prompt(self, query_txt: str):
        '''
        Create LLM prompt, with structure: <problem description> <delimiter> <query text>
        :return:
        '''
        delim = self._delimiter + '\n\n' if self._delimiter else ''
        query = f'{self._txt_desc}: {query_txt}\n{self._cls_desc}: '
        prompt = f'{self._task_desc}\n\n{delim}{query}'
        return prompt

    def class_from_output(self, output: str):
        ''' Maps LLM's output to a class. '''
        output = label_normalize(output)
        for cls in self._classes:
            if output == label_normalize(cls): return cls
        if self._default_cls:
            return self._default_cls
        else:
            return self._classes[0]

def test_template_simple():
    t = LLMZeroShotTemplate(
        task_id='tweet_senti_zero-shot',
        cls_defs=['positive', 'negative', 'neutral'],
        txt_desc='Tweet',
        class_desc='Sentiment',
        default_cls='neutral',
    )
    pr = t.create_prompt(query_txt='Have seen better works')
    print_template_prompt(t, pr)

def test_template_complex():
    t = LLMZeroShotTemplate(
        task_id='topic_classif',
        cls_defs=[('sport', ''), ('politics', 'political news'),
                  ('world', 'news from out of the US'), ('stars', 'life of famous persons')],
        txt_desc='News text',
        class_desc='Topic',
        delimiter='---',
        default_cls='politics',
    )
    pr = t.create_prompt(query_txt='Major industry players support the incumbent president')
    print_template_prompt(t, pr)

def test_template_complex_custom():
    t = LLMZeroShotTemplate(
        task_id='topic_classif',
        task_desc='Determine whether the topic of the news text is about sport, politics, world (world news) or stars (famous people)',
        cls_defs=['sport', 'politics', 'world', 'stars'],
        txt_desc='news text',
        class_desc='topic',
        delimiter='---',
        default_cls='politics',
    )
    pr = t.create_prompt(query_txt='Major industry players support the incumbent president')
    print_template_prompt(t, pr)

def print_template_prompt(t, pr):
    print(f'ID: {t.id}', '\n')
    print('PROMPT:')
    print(pr)


if __name__ == '__main__':
    test_template_simple()
    #test_template_complex()
    #test_template_complex_custom()
