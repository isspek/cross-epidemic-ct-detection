from pathlib import Path
from code.classification.eval import precision_recall_fscore_support
import pandas as pd
from collections import Counter

def most_common(lst):
    data = Counter(lst)
    return data.most_common(1)[0][0]

diseases=['zika', 'covid', 'monkeypox']
data_dir=Path('data/')

for disease in diseases:
    data = pd.read_csv(data_dir/f'{disease}/data.tsv', sep='\t')
    true_values = data['label'].tolist()
    pred = most_common(true_values)
    preds =  [pred for i in  range(len(true_values))]
    print(disease)
    print('majority class prediction')
    print(precision_recall_fscore_support(true_values, preds, average='macro'))