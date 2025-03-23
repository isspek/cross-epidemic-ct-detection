import hashlib
import regex as re

def str_hash(s: str) -> str:
    '''
    Creat a good PERMANENT hash of a string, with a minuscule probability of clashes.
    Permanency means this is a pure function, depending only on the input string.
    '''
    h = hashlib.sha256(s.encode()).hexdigest()
    return f'{h}'

def extract_first_word(s: str) -> str:
    ''' If string is not a pure alphanumeric sequence, extract the first word from a string -
    everything before a whitespace or an interpunction mark. '''
    if re.match(r'^[a-zA-Z0-9]+$', s): return s
    return re.match(r'^[a-zA-Z0-9]+', s).group(0)

def label_normalize(out: str) -> str:
    '''
    Turns the output of an LLM, as well as class labels to a canonic form so that they can be matched.
    Lowercse, remove whitespace and non-alphanumerics from both borders.
    :return:
    '''
    out = out.strip().lower()
    remalnum = lambda s: re.sub(r'^[^a-zA-Z0-9]*|[^a-zA-Z0-9]*$', '', s)
    out = remalnum(out)
    return out