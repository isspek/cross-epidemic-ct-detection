from typing import Tuple

import openai

from llmag.llm.LLCompletionModel import LLCompletionModel

class GptLLMCompletionModel(LLCompletionModel):
    '''
    Interface for OpenAI's text completion model.
    '''

    def __init__(self, model_id):
        '''
        :param model_id: OpenAI model identifier, ex. 'text-davinci-002'
        '''
        id = f'{self.__class__.__name__}_model[{model_id}]'
        self._model_id = model_id
        super().__init__(id=id)

    def __call__(self, prompt: str, **params) -> Tuple[str, object]:
        '''
        :param params: model config params that will be passed to openai.Completion.create()
        :return:
        '''
        response = openai.Completion.create(model=self._model_id, prompt=prompt, **params)
        txt_answer = response['choices'][0]['text'].strip()
        return txt_answer, response

def test_model(model_id='text-davinci-002'):
    from llmag.llm.openai_llm.openai_utils import set_api_key
    set_api_key()
    model = GptLLMCompletionModel(model_id)
    txt_answ, response = model('Good day to you, my dear AI!')
    print('Answer: ', txt_answ)
    print('Response:\n', response)

if __name__ == '__main__':
    test_model()