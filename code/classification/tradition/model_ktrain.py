import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

logger.add(f"{__name__}.log", rotation="500 MB")

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    random.seed(random_seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_diseases", type=lambda s: [i for i in s.split(',')])
    parser.add_argument("--test_disease", type=str)
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--class_weight", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--disease", type=str)
    parser.add_argument("--predictions_dir", type=str)

    args = parser.parse_args()
    set_random_seed(args.random_seed)

    base_model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    best_f1 = -1
    best_model = None
    best_params = None

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "logreg__C": [0.1, 1, 10],
        "logreg__class_weight": ["balanced"]
    }

    if args.train:
        train_dataset = pd.read_csv(args.train_dataset, sep='\t')
        dev_dataset = pd.read_csv(args.val_dataset, sep='\t')
        X_train = train_dataset['text'].tolist()
        y_train = train_dataset['label'].tolist()
        best_f1 = -1
        best_model = None
        best_params = None
        for ngram in param_grid["tfidf__ngram_range"]:
            for C in param_grid["logreg__C"]:
                for cw in param_grid["logreg__class_weight"]:

                    model = Pipeline([
                        ("tfidf", TfidfVectorizer(ngram_range=ngram)),
                        ("logreg", LogisticRegression(C=C, class_weight=cw, max_iter=500))
                    ])

                    # Train on train set only
                    model.fit(X_train, y_train)

                    # Predict on dev set
                    X_dev = dev_dataset['text'].tolist()
                    y_dev = dev_dataset['label'].tolist()
                    dev_pred = model.predict(X_dev)
                    f1 = f1_score(y_dev, dev_pred, average="macro")

                    print(f"Params={ngram, C, cw} → Dev F1 = {f1:.4f}")

                    # Track the best dev F1
                    if f1 > best_f1:
                        best_f1 = f1
                        best_model = model
                        best_params = (ngram, C, cw)

    if args.eval:
        model_id = args.model_id
        test_disease = args.test_disease
        test_dataset = args.test_dataset
        model_path = args.model_path
        test_data = pd.read_csv(test_dataset, sep='\t')
        logger.info(f'{model_path} is loading')
        result_output = f"{args.predictions_dir}/results.tsv"

        logger.info(f'Saving the predictions to {result_output}')

        logger.info(f'Model Path {model_path}')
        texts = test_data['text'].tolist()
        preds = best_model.predict(texts)

        predictions = []
        for text, pred in zip(texts, preds):
            predictions.append({
                'text': text,
                'label': pred,
                'id_str': None
            })
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(result_output, index=False, sep='\t')