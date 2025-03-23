import time
from enum import Enum
from pathlib import Path

import langchain as lc
from huggingface_hub.utils import HfHubHTTPError
from langchain import PromptTemplate, OpenAI
from langchain.llms import BaseLLM
from langchain_huggingface import HuggingFaceEndpoint

from code.classification.llm.llm_setup import langchain_sqlite_cache
from code.classification.llm.sagemaker_utils import create_sagemaker_langchain_llm
from code.corpora.data_loaders import load_excel_dset
from code.pysettings import OPENAI_API_KEY, ZIKA_LABELED_DATASET, ZERO_SHOT_LABELING_ROOT, FLAN_T5_XXL_ENDPOINT, \
    LLAMA_65B_ENDPOINT
from code.llmag.llmag.llmag_utils.misc_utils import label_normalize, extract_first_word


def get_cached_openai_llm(model_name='gpt-3.5-turbo-0301'):
    lc.llm_cache = langchain_sqlite_cache()
    llm = OpenAI(model_name=model_name, temperature=0, max_tokens=5, n=1, openai_api_key=OPENAI_API_KEY)
    return llm

CONSPIRACY_PROMPT_ZERO_SHOT = \
'''Determine weather the {text_desc} supports a conspiracy theory about {disease_desc}. Answer yes or no.

{text_desc}: {text}
supports a conspiracy (YES or NO): '''

# prompt based on the definition of the conspiracy theory used by the annotators
CONSPIRACY_PROMPT_ZERO_SHOT_DEFINITION = \
'''
Here is the precise definition of what 'supporting a conspiracy theory' means:
--- START OF DEFINITION ---
A post explicitly supports a conspiracy theory when it claims that a secret
agent or group with a hidden agenda is behind an event, usually with malicious intent. 
It is not necessary that every element is present explicitly in the text, but only some of them. 
These are examples posts supporting a conspiracy theory (with the element in parentheses): 
"Bill Gates is behind all of this" (agent), 
"Of course this massive vaccination is done to plan mind-control microchips into people" (agenda), 
"Those awake know that the endgame of this whole charade is a new world order" (intent).

A post implicitly supports a conspiracy theory if it hints at it, for example by using the word ‘plandemic’. 
The problem with detection of such conspiracy claims is that one needs to know the 
hints such as short claims and keywords, which can require knowing about the broader context. 
If an broader context is referenced, use your knowledge in order to understand the reference.
--- END OF DEFINITION ---

Determine, based on the provided definition, weather the {text_desc} supports a conspiracy theory about {disease_desc}. Answer yes or no.

{text_desc}: {text}
supports a conspiracy (YES or NO): '''

# original annotator's instructions
'''
- A post is considered that support explicitly a conspiracy theory when claims that a secret
agent/group with a hidden agenda is behind an event, usually with malicious intent. It is not
necessary that every element was present explicitly in the text but yes some of them. For
example: "Bill Gates is behind all of this" (agent), "Of course this massive vaccination is done to
plan mind-control microchips into people" (agenda), "Those awake know that the endgame of
this whole charade is a new world order" (intent).
- A post is considered to support implicitly a conspiracy theory in a supportive manner, for
example by using the word ‘plandemic’. Problem with detection of these conspiracy claims
is that one needs to know the hints such as short claims and keywords, which can require
searching other tweets for context. If an outside context is referenced, such as a person or
event - search the web in order to gain understanding of the reference

'''

class Disease(Enum):
    ZIKA = 1
    EBOLA = 2
    COVID = 3
    MONKEYPOX = 4

# map of disease to description (for the prompt)
DISEASE_DESC = {
    Disease.ZIKA: 'Zika',
    Disease.EBOLA: 'Ebola',
    Disease.COVID: 'COVID-19',
    Disease.MONKEYPOX: 'Monkeypox'
}

DISEASE_LABEL = {
    Disease.ZIKA: 'zika',
    Disease.EBOLA: 'ebola',
    Disease.COVID: 'covid',
    Disease.MONKEYPOX: 'monkeypox'
}

class LabelZeroShotClassif():
    '''
    Classifies, using a LLM and based on a 'category label', weather a text is encompassed by the definition.
    Label is a short phrase-like description of the category.
    '''

    def __init__(self, llm: BaseLLM, prompt: str, categ_label: str, disease_desc: str,
                 text_desc: str = 'tweet', verbose=False):
        self._llm = llm
        self._prompt = PromptTemplate.from_template(prompt)
        self._disease_desc = disease_desc
        self._text_desc = text_desc
        self._verbose = verbose

    def _label_from_output(self, output):
        return output.strip().lower()

    def __call__(self, text) -> int:
        prompt = self._prompt.format_prompt(text=text, text_desc=self._text_desc, disease_desc=self._disease_desc)
        if self._verbose: print(f'PROMPT: {prompt.text}')
        res = self._llm(prompt.text)
        result_normalized = label_normalize(res)
        if self._verbose: print(f'RAW RESULT: [{result_normalized}]')
        # if the result_normalized does not contain any alphabetic characters, treat it as 'no'
        if not any(c.isalpha() for c in result_normalized):
            print(f'WARNING: no alphabetic characters in the result, treating as "no"')
            if not self._verbose:
                print(f'INPUT: [{text}]')
                print(f'RAW RESULT: [{result_normalized}]')
            result_normalized = 'no'
        res = extract_first_word(result_normalized)
        if self._verbose: print(f'RESULT: {res}\n')
        return 1 if res == 'yes' else 0

def get_prompt(defintion=False):
    return CONSPIRACY_PROMPT_ZERO_SHOT_DEFINITION if defintion else CONSPIRACY_PROMPT_ZERO_SHOT

def create_conspiracy_def_zero_shot_gpt(disease: Disease, model='gpt-3.5-turbo-0301', definition=False):
    ''' Factory method for zero-shot classifiers. '''
    llm = get_cached_openai_llm(model)
    prompt = get_prompt(definition)
    text_desc = 'tweet'
    return LabelZeroShotClassif(llm, prompt, categ_label='conspiracy', disease_desc=DISEASE_DESC[disease], text_desc=text_desc)

def create_huggingface_llm(model=None, wait_time=30, retries=3):
    '''
    Configure and create a langchain-compatible text-generation LLM based on a Hugging Face model.
    :param model: a label of one of the used models: flan-t5
            this ensures that the model is 'supported', ie, configured correctly
    :return:
    '''
    API_TOKEN = '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'
    if model == 'flan-t5': model_id = "google/flan-t5-xxl"
    elif model == 'mistral': model_id = "mistralai/Mistral-7B-Instruct-v0.3"
    elif model == 'llama': model_id = 'meta-llama/Meta-Llama-3-70B-Instruct'
    else: raise ValueError(f'Unsupported HF model: {model}')
    while True:
        try:
            llm = HuggingFaceEndpoint(
                repo_id=model_id,
                task="text-generation",
                max_new_tokens=5,
                do_sample=False,
                huggingfacehub_api_token = API_TOKEN
            )
            res = llm.invoke("Hugging Face is") # test the model
            #print(f'HF result:\n[{res}]')
            #print(f'LLM type: [{type(llm)}]')
            return llm
        except Exception as e:
            if retries == 0:
                print(f'ERROR: failed to create Hugging Face model {model_id}:\n{e}')
                return None
            print(f'WARNING: timeout for the model {model_id}:\n{e}')
            print(f'WAITING FOR {wait_time} seconds before retry...')
            time.sleep(wait_time)
            retries -= 1

def label_disease_dataset_zero_shot(disease: Disease, dset_path: str, text_col, model='gpt-3.5-turbo-0301',
                                    test_only=None, endpoint_name=None, definition=False, verbose=True):
    ''' Label a dataset using a zero-shot classifier, add the label to the dataset save it as tsv. '''
    if 'zero-shot' in str(Path(dset_path).name): # ignore already labeled datasets
        print(f'Ignoring {dset_path}')
        return
    model = model.strip().lower()
    if model.startswith('sagemaker'): llm = create_sagemaker_langchain_llm(endpoint_name)
    elif model.startswith('gpt'): llm = get_cached_openai_llm(model)
    elif model.startswith('hf-'): llm = create_huggingface_llm(model[3:], retries=0)
    # build the classifier
    prompt = get_prompt(definition)
    classifier = LabelZeroShotClassif(llm, prompt, disease_desc=DISEASE_DESC[disease],
                                        categ_label='conspiracy', text_desc='tweet', verbose=verbose)
    ext = Path(dset_path).suffix
    #sep= '\t' if ext == '.tsv' else ',' if ext == '.csv' else None
    # do try catch loading the dataset, first with ',' then with '\t' as the separator, if both fail raise exception
    df = None
    for sep in [',', '\t']:
        try: df = load_excel_dset(dset_path, sep=sep)
        except: continue
        else: break
    if df is None:
        print(f'Warning: failed to load dataset {dset_path}')
        return
    #df = load_excel_dset(dset_path, sep=sep)
    N = len(df)
    cnt = test_only if test_only else -1
    # iterate over the dataframe rows, classify each text and add the label to the dataframe
    def_label = 'def' if definition else 'no-def'
    column_name = f'{model}-{def_label}-zero-shot'
    print(f'Starting zero-shot labeling for dataset {dset_path}')
    for i, row in df.iterrows():
        text = row[text_col]
        label = classifier(text)
        #print(f'{i}/{N}: {text} -> {label}')
        if (i+1) % 20 == 0: print(f'labeled {i+1}/{N}')
        df.loc[i, column_name] = label
        cnt -= 1
        if cnt == 0: break
    dset_path = dset_path.replace(f'{ext}', f'_zero-shot{ext}')
    df.to_csv(dset_path, sep=sep, index=False)

def test_zero_shot_labeling():
    label_disease_dataset_zero_shot(Disease.ZIKA, ZIKA_LABELED_DATASET, 'full_text', test_only=5)

def run_all_corpora_zero_shot(test_only=None, model='gpt-3.5-turbo-0301', definition=False, verbose=True):
    # for disease in [Disease.COVID, Disease.MONKEYPOX, Disease.EBOLA, Disease.ZIKA]:
    for disease in [Disease.ZIKA]:
        for typ in ['csv', 'tsv']:
            for dset in (Path(ZERO_SHOT_LABELING_ROOT)/DISEASE_LABEL[disease]).glob(f'*.{typ}'):
                print(f'LABELING: {str(dset)}')
                label_disease_dataset_zero_shot(disease, str(dset), text_col='text', test_only=test_only,
                                                model=model, endpoint_name=LLAMA_65B_ENDPOINT,
                                                definition=definition, verbose=verbose)

if __name__ == '__main__':
    #test_zero_shot_labeling()
    #run_all_corpora_zero_shot(test_only=20, model='hf-llama', definition=True, verbose=False)
    # run_all_corpora_zero_shot(test_only=None, model='gpt-3.5-turbo-0301', definition=True, verbose=False)
    run_all_corpora_zero_shot(test_only=None, model='gpt-4o', definition=True, verbose=False)
    #create_conspiracy_def_zero_shot_huggingface(Disease.ZIKA, model='mistral')