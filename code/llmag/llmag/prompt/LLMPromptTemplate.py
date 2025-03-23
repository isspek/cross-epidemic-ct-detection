from abc import abstractmethod

from llmag.common.Identifiable import Identifiable
from llmag.llmag_utils.misc_utils import str_hash


class LLMPromptTemplate(Identifiable):
    '''
    Base class for creation prompt-based queries to LLMs.
    Subclasses share identifiability that includes identity of the strings used to construct the prompt,
    and a method to construct the prompt text, based on parameters such as query text.
    Some example subtasks are: few- and zero-shot classification, paraphrasing or summarization of text,
    or any operation that gives a task to LLM in the format of the text prompt composed of
    class params, and run-time params such as texts (few shot examples, texts to summarize, etc.)
    '''

    def __init__(self, task_id, task_desc):
        '''
        ID-composing (structural) member variables should be initialized only once, in the constructor.
        :param task_id: id of the LLM task
        :param task_desc: description of the LLM task
        '''
        self._id_vars = []
        self._task_id = task_id; self._idable('_task_id')
        self._task_desc = task_desc; self._idable('_task_desc')

    @property
    def id(self):
        if not hasattr(self, '_id'): self._compose_id()
        return self._id

    @abstractmethod
    def create_prompt(self, *vars, **params) -> str: ...

    def _idable(self, var_name: str):
        '''
        Add the variable to the list of variables that define the identity of the object.
        These should be the variables (convertable to string) that define the text of the prompt,
        as well as variables influencing the structure of the prompt.
        Subclasses should call this method to register such variables, to automatize ID-ability.
        :param var_name: name of the object's member variable
        :return:
        '''
        self._id_vars.append(var_name)

    def _compose_id(self):
        '''
        Identity of the class is defined by task_id and all the relevant variables.
        :return:
        '''
        self._id_vars.sort()
        idvar_vals = "-".join([f"{v}={str(getattr(self, v))}" for v in self._id_vars])
        self._id = f'{self._task_id}-{self.__class__.__name__}-{str_hash(idvar_vals)}'

def test_template():
    # comment @abstractclass above for this to run
    t = LLMPromptTemplate(task_id='test_task', task_desc='just a testing task')
    print(t.id)

if __name__ == '__main__':
    test_template()