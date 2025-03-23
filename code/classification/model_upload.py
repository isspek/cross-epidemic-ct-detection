import os
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, TrainingArguments, Trainer, TextClassificationPipeline,
                          EvalPrediction, EarlyStoppingCallback)
from huggingface_hub import HfApi, HfFolder
from argparse import ArgumentParser
from pathlib import Path
import shutil

if __name__ == '__main__':
    # parser = ArgumentParser()
    # parser.add_argument('--pretrained_model')
    # parser.add_argument('--model_path')
    # args = parser.parse_args()
    #
    # pretrained_model = args.pretrained_model
    # model_path = Path(args.model_path)

    config = AutoConfig.from_pretrained(
        pretrained_model,
        num_labels=2,
        id2label={0: "0", 1: "1"},
        label2id={"1": 1, "0": 0},
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        config=config,
        ignore_mismatched_sizes=True
    )

    model.push_to_hub(f"isspek/{model_path.name}")
    shutil.rmtree(model_path)
