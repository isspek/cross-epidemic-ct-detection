# Cross Epidemic - Medical Conspiracy Detection

This repository contains the source code for the paper: "Detection of Medical Conspiracy Theories with Limited
Resources: Using Data from Prior Epidemics and LLMs".

## Virtual Environment

```shell
conda create -n conspiracy python=3.8
```

Activate the virtual environment:
```shell
conda activate conspiracy
```

Install the required libraries:
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
```shell
pip install -r requirements.txt
```

## Experiments

Make sure that the labeled datasets are under the `data` folder. We will do our analysis on the conspiracy labeled
samples.

### Dataset Analysis
The code for the dataset analysis (Figure 3 in the paper) at `code/analysis/content/visualize_diff`. The visualizations are also at `results` in the same folder.

### Fine-tuning
We fine-tune BERT, XLNet and RoBERTa models.

Example command is as follows:
```bash
screen -L -Logfile bert_ktrain.log scripts/bert_ktrain.sh
```
The scripts are at `scripts`.
The trained models are already hosted in Huggingface, you don't need to retrain. You can just change the model_path from models/${model_id} to isspek/${model_id}.
You just need to remove --train from the commands, so it does not retrain the model, use for only inferencing.
Add _weight to the model name for the weighted models whose results are given in Appendix.

### LLM Experiments

The related source codes are at `code/classification/llm`.
Create a pysettings.py, fill it with the required parameters/keys.

```shell
python -m code.classification.llm
```

## Dataset
The anonymized dataset (Zika, Monkeypox, Ebola) is at `dataset`. Due to the X license, we only share the IDs.
The list of search queries from the related literature at `data/search_queries`.

For the other datasets, please follow the instructions of the dataset owners.
The other Conspiracy Theory datasets:
- [COCO](https://link.springer.com/article/10.1007/s42001-023-00200-3)
- [Non-epidemic](https://dl.acm.org/doi/10.1145/3487553.3524665)

Send an e-mail to `ibarsch@doctor.upv.es` for other requests about the dataset.

## Citation
The citation infomration of our paper:
```bibtex
TODO
```
