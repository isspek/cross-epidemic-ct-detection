'''
Initializes the framework with the key (upon import) and contains utility methods.
'''

import openai

def set_api_key(key=None):
    if key is None:
        from llmag.settings import OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY
    else:
        openai.api_key = key

