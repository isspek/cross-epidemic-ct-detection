from functools import partial

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

from llmag.caching.cache_function import CachedFunction

f1_macro = partial(f1_score, average='macro')
recall_macro = partial(recall_score, average='macro')
prec_macro = partial(precision_score, average='macro')

from llmag.classify.LLMZeroShotClassifier import LLMZeroShotClassifer
from llmag.corpus.DfCorpus import DfCorpus
from llmag.llm.openai_llm.GptChatAsCompletionLLM import GptChatAsCompletionLLM
from llmag.llm.openai_llm.openai_utils import set_api_key

from code.pysettings import ZIKA_LABELED_DATASET, LLM_CACHE_FOLDER

from code.pysettings import OPENAI_API_KEY
from code.corpora.data_loaders import load_excel_dset
from llmag.prompt.LLMZeroShotTemplate import LLMZeroShotTemplate

ZIKA_DSET = '/datafast/corpora/medical_conspiracies/labeled/zika_data_samples_1000.tsv'

def load_zika_corpus(file_path, filter_labeled=True):
    df = load_excel_dset(file_path, sep='\t')
    df['index'] = df.index # workaround until the id column is fixed to contain unique values
    # keep only rows with non-empty 'conspiracy' column
    if filter_labeled:
        df = df[df['conspiracy'].notna()]
    return DfCorpus(file_path, df, id_col='index', txt_col='cleaned_text', properties=True)

def zero_shot_classif_demo(test_size=30, verbose=True, caching=True):
    corpus = load_zika_corpus(ZIKA_LABELED_DATASET)
    # select a random subset of examples for testing
    ids = corpus.text_ids()
    cls = [ corpus[id].conspiracy for id in ids ]
    _, test_id_cls = train_test_split(ids, test_size=test_size, stratify=cls, random_state=123450)
    #for id in ids: print('conspiracy:', corpus[id].conspiracy, 'text:', corpus[id].text)
    # setup classifier
    t = LLMZeroShotTemplate(
        task_id='zika_zero-shot',
        task_desc='Determine weather the tweet supports a conspiracy theory about Zika - yes or no.',
        cls_defs=['yes', 'no'],
        txt_desc='tweet',
        class_desc='supports a conspiracy',
        default_cls='no',
    )
    if verbose:
        print('TEMPLATE:')
        print(t.create_prompt('just a sample text'))
        print()
    # setup classification
    llm = GptChatAsCompletionLLM(model_id='gpt-3.5-turbo-0301')
    classifier = LLMZeroShotClassifer(llm, t, corpus)
    if caching:
        classifier = CachedFunction(classifier, cacheFolder=LLM_CACHE_FOLDER, saveEvery=1, verbose=False)
    # classify
    truth = [corpus[id].conspiracy for id in test_id_cls]
    pred = []; answ2label = { 'yes': 1, 'no': 0 }
    for id in test_id_cls:
        cls_answ, raw_answ, _, prompt = classifier(id)
        pred_label = answ2label[cls_answ]
        pred.append(pred_label)
        if verbose:
            print(f'TEXT: {corpus[id].text}')
            print(f'TRUE_CLASS: {corpus[id].conspiracy}')
            print()
            print(f'PROMPT: {prompt}')
            print()
            print(f'CLASS: {cls_answ}, RAW_ANSWER: {raw_answ}')
            print(f'PRED. LABEL: {pred_label}, MATCH: {pred_label == corpus[id].conspiracy}')
            print('\n')
    print(f'true: {truth}')
    print(f'pred: {pred}')
    print()
    print(f'f1: {f1_score(truth, pred):.3f}')
    print(f'precision: {precision_score(truth, pred):.3f}')
    print(f'recall: {recall_score(truth, pred):.3f}')
    print(f'accuracy: {accuracy_score(truth, pred):.3f}')
    #print(f'recall: {recall_macro(pred, truth):.3f}')

if __name__ == '__main__':
    set_api_key(OPENAI_API_KEY)
    #load_zika_corpus(ZIKA_LABELED_DATASET)
    zero_shot_classif_demo(test_size=500)