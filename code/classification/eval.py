import chardet
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support, f1_score, classification_report

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--labels')
    parser.add_argument('--preds')

    args = parser.parse_args()

    labels = args.labels

    labels = pd.read_csv(args.labels, sep='\t')

    # with open(args.preds, 'rb') as f:
    #     enc = chardet.detect(f.read())  # or readline if the file is large
    # preds = pd.read_csv(args.preds,encoding=enc['encoding'], sep='\t')

    preds = pd.read_csv(args.preds, sep='\t')

    if 'label' not in preds.columns:
        preds = pd.read_csv(args.preds, sep=',')

    preds['label'] = preds['label'].apply(lambda x: 0 if x == "-1" else x)

    print(len(labels))

    print(len(preds))

    print("Non-labeled samples")

    preds.rename(columns={'full_text': 'text'}, inplace=True)
    preds.rename(columns={'id_str': 'id'}, inplace=True)

    y_pred = []
    y_true = []
    for idx, row in preds.iterrows():
        y_true.append(labels[labels['text'] == row['text']]['label'].tolist()[0])
        y_pred.append(row['label'])

    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f1_ct = f1_score(y_true, y_pred, average='macro')
    print('==macro==')
    print(f"{precision} {recall} {f1_macro} {f1_ct}")

    precision, recall, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    f1_ct = f1_score(y_true, y_pred, average='binary')
    print('==binary==')
    print(f"{precision} {recall} {f1_macro} {f1_ct}")

    print('==Classification Report===')
    print(classification_report(y_true=y_true, y_pred=y_pred))