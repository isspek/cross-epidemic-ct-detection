import random
import numpy as np
import torch

# copied from code.claim_extraction.clustering.set_random_seed
# to avoid db initialization on import
# TODO replace other uses with this version
# TODO move to a generic utils module
def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def dframe_info(df):
    print('COLUMNS')
    for c in df: print(c)
    print(df)