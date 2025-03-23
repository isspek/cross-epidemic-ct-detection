from time import sleep
from typing import Tuple
import openai

from llmag.llm.LLCompletionModel import LLCompletionModel

class GptChatAsCompletionLLM(LLCompletionModel):
    '''
    Gpt chat model for use a simple completion (text -> text) model.
    '''

    def __init__(self, model_id, wait_ms=20):
        '''
        :param model_id: OpenAI chat model identifier, ex. 'gpt-3.5-turbo-0301'
        '''
        id = f'{self.__class__.__name__}_model[{model_id}]'
        self._model_id = model_id
        self._wait_ms = wait_ms
        super().__init__(id=id)

    def __call__(self, prompt: str, **params) -> Tuple[str, object]:
        '''
        :param params: model config params that will be passed to openai.ChatCompletion.create()
        :return:
        '''
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."}, # ignore for now, plus gpt-3.5-turbo-0301 doesn't support this
            {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(model=self._model_id, messages=messages, **params)
        if self._wait_ms: sleep(self._wait_ms/1000)
        txt_answer = response['choices'][0]['message']['content'].strip()
        return txt_answer, response

def test_model(model_id='gpt-3.5-turbo-0301'):
    from llmag.llm.openai_llm.openai_utils import set_api_key
    set_api_key()
    model = GptChatAsCompletionLLM(model_id)
    txt_answ, response = model('Good day to you, my dear AI!')
    print('Answer: ', txt_answ)
    print('Response:\n', response)

if __name__ == '__main__':
    #test_model()
    test_model(model_id='gpt-4')
