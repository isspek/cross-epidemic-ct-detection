import chardet
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--preds')
    parser.add_argument('--fold_data')


    args = parser.parse_args()

    fold_data = args.fold_data
    fold_data = pd.read_csv(args.fold_data, sep='\t')

    print(len(fold_data['id_str'].tolist()))

    data = pd.read_csv(args.preds, sep='\t')
    data = data[data['id_str'].isin(fold_data['id_str'].tolist())]

    annots = ['annot_1', 'annot_2', 'annot_3']

    for annot in annots:
        print(f"Checking {annot}...")
        print("Majority")
        majority_labels = data['label'].tolist()
        y_pred = data[annot].tolist()

        precision, recall, f1,_ = precision_recall_fscore_support(y_true=majority_labels, y_pred=y_pred, average=None, labels=[0,1])
        print('precision')
        print(precision)

        print('recall')
        print(recall)

        print('f1')
        print(f1)

        for other_annot in annots:
            if other_annot == annot:
                continue
            print("Other label")
            print(other_annot)
            other_labels = data[other_annot].tolist()
            y_pred = data[annot].tolist()
            precision, recall, f1,_ = precision_recall_fscore_support(y_true=other_labels, y_pred=y_pred, average=None, labels=[0,1])

            print('precision')
            print(precision)

            print('recall')
            print(recall)

            print('f1')
            print(f1)