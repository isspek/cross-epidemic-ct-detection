#!/bin/bash
#SBATCH --time=36:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1

pretrained_model=roberta-base
model_name=roberta-base

train_batch=16
num_epochs=100

cuda_device=-1
learning_rate=2e-5

random_seed=0

# for coredisease in ebola zika monkeypox;
for coredisease in covid;
do
 for disease in ${coredisease} ${coredisease}_llama ${coredisease}_chatgpt ${coredisease}_mistral ${coredisease}_top3 ${coredisease}_gpt4o;
 do

 echo $disease training and testing
 train_diseases=$disease
 test_disease=${coredisease}

for undersampling in 0.1 0.2 0.3 0.4 0.5;
do
    for n_fold in 1 2 3 4 5;
    # for n_fold in 3;
     do
     n_fold=$n_fold

     model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}_undersampling_${undersampling}
     echo training $model_id
     predictions_dir=data/results/${model_id}/${test_disease}
     model_path=models/${model_id}
     train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train_${undersampling}.tsv

     echo $train_dataset

     val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
     test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

     echo $model_id
     echo Train disease $train_diseases
     echo Test disease $test_diseases

     mkdir -p $predictions_dir

     python3 -m code.classification.transformers.model_ktrain \
     --cuda_device $cuda_device \
     --pretrained_model $pretrained_model \
     --random_seed $random_seed \
     --learning_rate $learning_rate \
     --train_batch $train_batch \
     --model_path $model_path \
     --num_epochs $num_epochs \
     --eval \
     --train_dataset $train_dataset \
     --test_dataset $test_dataset \
     --val_dataset $val_dataset \
     --predictions_dir $predictions_dir \
     --train_diseases $train_diseases \
     --test_disease $test_disease \
     --model normal \
     --model_id $model_id \
     --train


 echo $disease
     echo $n_fold results:
     python3 -m code.classification.eval \
     --labels ${test_dataset} \
     --preds ${predictions_dir}/results.tsv

     done
 done
 done
done