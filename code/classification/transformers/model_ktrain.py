import numpy as np
import os
import pandas as pd
import random
import torch
from argparse import ArgumentParser
from loguru import logger
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, TrainingArguments, Trainer, TextClassificationPipeline,
                          EvalPrediction, EarlyStoppingCallback)
from transformers import pipeline

logger.add(f"{__name__}.log", rotation="500 MB")
os.environ["WANDB_DISABLED"] = "true"


class ConspiracyDataset(Dataset):
    def __init__(self, data, tokenizer, source_max_length: int = 128):
        self.tokenizer = tokenizer
        self.source_max_length = source_max_length
        self.data = data
        print(data['label'].value_counts())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        input_text = sample["text"]
        label = sample["label"]
        source_encodings = self.tokenizer.batch_encode_plus([input_text], max_length=self.source_max_length,
                                                            pad_to_max_length=True, truncation=True,
                                                            padding="max_length", return_tensors='pt',
                                                            return_token_type_ids=False)

        return dict(
            input_ids=source_encodings['input_ids'].squeeze(0),
            attention_mask=source_encodings['attention_mask'].squeeze(0),
            labels=torch.LongTensor([label]),
        )


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


class CustomTrainer(Trainer):

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor(self.weights))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_accuracy(p: EvalPrediction):
    labels = p.label_ids
    preds = np.argmax(p.predictions, axis=1)
    map_weighted = average_precision_score(
        y_true=labels, y_score=preds, average='weighted')
    map_macro = average_precision_score(y_true=labels, y_score=preds, average='macro')
    f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {"map_weighted": map_weighted, "map_macro": map_macro,
            "f1-score": f1}


class TransformerModel:

    def __init__(self, pretrained_model):
        self.pretrained_model = pretrained_model

    def train(self, train_data,
              val_data,
              learning_rate,
              train_batch,
              num_epochs,
              device,
              random_seed,
              model_path,
              class_weight=None):
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

        train_dataset = ConspiracyDataset(
            data=train_data,
            tokenizer=tokenizer
        )

        val_dataset = ConspiracyDataset(
            data=val_data,
            tokenizer=tokenizer
        )

        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            num_labels=2
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            self.pretrained_model,
            config=config,
        )

        training_args = TrainingArguments(
            learning_rate=learning_rate,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch,
            output_dir=model_path,
            overwrite_output_dir=True,
            do_eval=True,
            do_train=True,
            remove_unused_columns=True,
            warmup_steps=len(train_data) // train_batch,
            save_strategy="steps",  # change it from step to epochs,
            evaluation_strategy="steps",
            logging_steps=500,
            save_total_limit=1,
            seed=random_seed,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1-score',
            disable_tqdm=False
        )

        if class_weight:
            _class_weight = torch.tensor([len(train_data[train_data["label"] == 1]) / len(train_data),
                                          len(train_data[train_data["label"] == 0]) / len(train_data)])
            _class_weight = _class_weight.to(torch.cuda.current_device())

            print(f"class_weight {_class_weight}")
            trainer = CustomTrainer(
                class_weights=_class_weight,
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_accuracy,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
            )
        else:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_accuracy,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
            )

        trainer.train()

        trainer.save_model(model_path)  # Saves the tokenizer too for easy upload

        del model
        del trainer
        torch.cuda.empty_cache()

    def test(self, test_data, model_path, result_output, random_seed):
        config = AutoConfig.from_pretrained(
            self.pretrained_model,
            num_labels=2,
            id2label={0: "0", 1: "1"},
            label2id={"1": 1, "0": 0},
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            config=config,
            ignore_mismatched_sizes=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model, max_length=128, truncation=True, padding=True)
        device = torch.cuda.current_device()
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)

        predictions = []

        for text in list(test_data["text"].unique()):
            result = pipe(text)[0]
            predictions.append({
                'text': text,
                'label': result["label"],
                'score': result["score"],
                'id_str': None
            })
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(result_output, index=False, sep='\t')
        del model


MODELS = {
    "normal": TransformerModel,
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--pretrained_model")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--train_batch", type=int)
    parser.add_argument("--train_diseases", type=lambda s: [i for i in s.split(',')])
    parser.add_argument("--test_disease", type=str)
    parser.add_argument("--train_dataset", type=str)
    parser.add_argument("--val_dataset", type=str)
    parser.add_argument("--test_dataset", type=str)
    parser.add_argument("--model_id", type=str)
    parser.add_argument("--random_seed", type=int)
    parser.add_argument("--few_shot", type=int)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--train_kfold", action='store_true')
    parser.add_argument("--eval_kfold", action='store_true')
    parser.add_argument("--class_weight", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--cuda_device", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--disease", type=str)
    parser.add_argument("--predictions_dir", type=str)
    parser.add_argument("--n_fold", type=int, default=10)

    args = parser.parse_args()

    cuda_device = args.cuda_device

    # if cuda_device != '-1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    # else:
    cuda_device = torch.cuda.current_device()

    n_gpus = torch.cuda.device_count()
    print(f"Number of gpu devices {n_gpus}")
    set_random_seed(args.random_seed)

    model = MODELS[args.model](pretrained_model=args.pretrained_model)

    if args.train:
        train_dataset = pd.read_csv(args.train_dataset, sep='\t')
        val_dataset = pd.read_csv(args.val_dataset, sep='\t')
        model.train(train_data=train_dataset,
                    val_data=val_dataset,
                    random_seed=args.random_seed,
                    device=args.cuda_device,
                    learning_rate=args.learning_rate,
                    num_epochs=args.num_epochs,
                    train_batch=args.train_batch,
                    model_path=args.model_path,
                    class_weight=args.class_weight)

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

        model.test(test_data=test_data,
                   random_seed=args.random_seed,
                   model_path=model_path,
                   result_output=result_output)
