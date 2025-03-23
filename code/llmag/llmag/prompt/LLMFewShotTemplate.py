from typing import List, Union, Tuple

from llmag.prompt.LLMZeroShotTemplate import LLMZeroShotTemplate, print_template_prompt


class LLMFewShotTemplate(LLMZeroShotTemplate):
    '''
    Template for LLM classification, creates prompt text from problem description, class labels,
    and labeled examples, and maps the LLM's output to labels.
    '''

    def __init__(self, task_id, cls_defs: Union[List[str], List[Tuple[str, str]]], task_desc=None,
                 txt_desc='text', class_desc='class', delimiter='\n', default_cls=None):
        '''
        :param task_id: id of the classification problem
        :param task_desc: description of the classification problem
        :param cls_defs: defines class labels, which should be short understandable phrases
                if descriptions are given, they will be used for the default problem description
        :param txt_desc: semantic description of texts in the prompt, ex. TWEET
        :param class_desc: semantic description of the class label in the prompt, ex: SENTIMENT
        :param delimiter: string to delimit the final query example from the rest
        :param default_cls: class to return if output cannot be matched to any class
        '''
        super().__init__(task_id=task_id, task_desc=task_desc, cls_defs=cls_defs, txt_desc=txt_desc,
                         class_desc=class_desc, delimiter=delimiter, default_cls=default_cls)

    def create_prompt(self, examples: List[Tuple[str, str]], query_txt: str):
        '''
        Create LLM prompt.
        :param examples: list of examples with class labels
        :param query_txt:
        :return:
        '''
        exs = [
            f'{self._txt_desc}: {ex[0]}\n{self._cls_desc}: {ex[1]}\n'
            for ex in examples
        ]
        exs = '\n'.join(exs)
        delim = self._delimiter+'\n\n' if self._delimiter else ''
        query = f'{self._txt_desc}: {query_txt}\n{self._cls_desc}: '
        prompt = f'{self._task_desc}\n\n{exs}\n{delim}{query}'
        return prompt

def test_template_simple():
    t = LLMFewShotTemplate(
        task_id='tweet_senti_few-shot',
        cls_defs=['positive', 'negative', 'neutral'],
        txt_desc='Tweet',
        class_desc='Sentiment',
        delimiter='---',
        default_cls='neutral',
    )
    pr = t.create_prompt(examples=[
        ('Excellent movie!', 'positive'),
        ('Complete garbage!', 'negative'),
        ('First movie of this director', 'neutral'),
        ],
        query_txt='Have seen better works'
    )
    print_template_prompt(t, pr)

def test_template_complex():
    t = LLMFewShotTemplate(
        task_id='topic_classif_few-shot',
        cls_defs=[('sport', ''), ('politics', 'political news'),
                  ('world', 'news from out of the US'), ('stars', 'life of famous persons')],
        txt_desc='News text',
        class_desc='Topic',
        delimiter='---',
        default_cls='politics',
    )
    pr = t.create_prompt(examples=[
        ('sport', 'Jordan returns to baseball'),
        ('sport', 'Superball playoffs to start'),
        ('stars', 'The top divorces of the year'),
        ('world', 'New EU-China trade deal signed'),
        ('politics', 'Primaries kick off tomorrow'),
        ('world', 'Tsunami hits east pacific'),
        ],
        query_txt='Major players support the incumbent'
    )
    print(pr)

def test_template_complex_custom():
    t = LLMFewShotTemplate(
        task_id='topic_classif_few-shot',
        task_desc='Decide weather a news text talks about the topic of either: '
                  'sport, politics, world (news of the world), or stars (famous people).',
        cls_defs=['sport', 'politics', 'world', 'stars'],
        txt_desc='news text',
        class_desc='topic',
        delimiter='---',
        default_cls='politics',
    )
    pr = t.create_prompt(examples=[
        ('sport', 'Jordan returns to baseball'),
        ('sport', 'Superball playoffs to start'),
        ('stars', 'The top divorces of the year'),
        ('world', 'New EU-China trade deal signed'),
        ('politics', 'Primaries kick off tomorrow'),
        ('world', 'Tsunami hits east pacific'),
        ],
        query_txt='Major players support the incumbent'
    )
    print(pr)

if __name__ == '__main__':
    #test_template_simple()
    #test_template_complex()
    test_template_complex_custom()
