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

# for disease in ebola zika covid monkeypox;
# do
# echo $disease training and testing
# train_diseases=$disease
# test_disease=$disease

#     for n_fold in 1 2 3 4 5;
#     do
#     n_fold=$n_fold

#     model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
#     echo training $model_id
#     predictions_dir=data/results/${model_id}/${test_disease}
#     model_path=models/${model_id}
#     train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
#     val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
#     test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

#     echo $model_id
#     echo Train disease $train_diseases
#     echo Test disease $test_diseases

#     mkdir -p $predictions_dir

#     python -m code.classification.transformers.model_ktrain \
#     --cuda_device $cuda_device \
#     --pretrained_model $pretrained_model \
#     --random_seed $random_seed \
#     --learning_rate $learning_rate \
#     --train_batch $train_batch \
#     --model_path $model_path \
#     --num_epochs $num_epochs \
#     --eval \
#     --train_dataset $train_dataset \
#     --test_dataset $test_dataset \
#     --val_dataset $val_dataset \
#     --predictions_dir $predictions_dir \
#     --train_diseases $train_diseases \
#     --test_disease $test_disease \
#     --model normal \
#     --train \
#     --model_id $model_id

# echo $disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

#     done
# done

# train_diseases=zika
# test_disease=covid
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# train_diseases=covid
# test_disease=monkeypox
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# train_diseases=non_epidemic_global_warning
# test_disease=covid
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# test_disease=monkeypox
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done


# train_diseases=zika_ebola
# test_disease=covid
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# train_diseases=covid_zika
# test_disease=monkeypox
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# train_diseases=covid_zika_ebola
# test_disease=monkeypox
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# test_disease=covid
# for train_diseases in zika_llama zika_mistral zika_chatgpt zika_gpt4o;
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done
# done

# test_disease=covid
# # for train_diseases in zika_llama_ebola_llama zika_mistral_ebola_mistral zika_chatgpt_ebola_chatgpt;
# for train_diseases in zika_chatgpt_ebola_chatgpt;
# do
# for n_fold in 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done
# done

# test_disease=monkeypox
# for train_diseases in covid_llama_zika_llama covid_mistral_zika_mistral covid_chatgpt_zika_chatgpt covid_gpt4o_zika_gpt4o;
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done
# done

# test_disease=monkeypox
# for train_diseases in covid_llama_zika_llama_ebola_llama covid_mistral_zika_mistral_ebola_mistral covid_chatgpt_zika_chatgpt_ebola_chatgpt covid_gpt4o_zika_gpt4o_ebola_gpt4o;
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done
# done

# test_disease=monkeypox
# for llm in llama mistral chatgpt gpt4o;
# do
# for train_diseases in covid_${llm};
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

# done
# done
# done



# test_disease=covid
# for llm in chatgpt gpt4o;
# do
# for train_diseases in zika_zika_conspi_users_${llm} zika_zika_conspi_users_${llm}_ebola_ebola_conspi_users_${llm};
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

# done
# done
# done


# test_disease=covid
# for llm in top3;
# do
# for train_diseases in zika_${llm}_ebola_${llm};
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

# done
# done
# done


# test_disease=monkeypox
# for llm in top3;
# do
# for train_diseases in covid_${llm}_zika_${llm} covid_${llm}_zika_${llm}_ebola_${llm};
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

# done
# done
# done

# for coredisease in monkeypox;
# do
#  for disease in ${coredisease} ${coredisease}_llama ${coredisease}_chatgpt ${coredisease}_mistral ${coredisease}_top3 ${coredisease}_gpt4o;
# do
# echo $disease training and testing
# train_diseases=$disease
# test_disease=$coredisease

#     for n_fold in 1 2 3 4 5;
#     do
#     n_fold=$n_fold

#     model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
#     echo training $model_id
#     predictions_dir=data/results/${model_id}/${test_disease}

#     # model_path=models/${model_id}
#     model_path=isspek/${model_id}
#     train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
#     val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
#     test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

#     echo $model_id
#     echo Train disease $train_diseases
#     echo Test disease $test_diseases

#     mkdir -p $predictions_dir

#     python3 -m code.classification.transformers.model_ktrain \
#     --cuda_device $cuda_device \
#     --pretrained_model $pretrained_model \
#     --random_seed $random_seed \
#     --learning_rate $learning_rate \
#     --train_batch $train_batch \
#     --model_path $model_path \
#     --num_epochs $num_epochs \
#     --eval \
#     --train_dataset $train_dataset \
#     --test_dataset $test_dataset \
#     --val_dataset $val_dataset \
#     --predictions_dir $predictions_dir \
#     --train_diseases $train_diseases \
#     --test_disease $test_disease \
#     --model normal \
#     --model_id $model_id \
#     --train

# echo $disease
#     echo $n_fold results:
#     python3 -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

#     done
# done
# done


# for coredisease in monkeypox;
# do
#  for disease in ${coredisease} ${coredisease}_llama ${coredisease}_chatgpt ${coredisease}_mistral ${coredisease}_top3 ${coredisease}_gpt4o;
# do
# echo $disease training and testing
# train_diseases=$disease
# test_disease=$coredisease

#     for n_fold in 1 2 3 4 5;
#     do
#     n_fold=$n_fold

#     model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}_weight
#     echo training $model_id
#     predictions_dir=data/results/${model_id}/${test_disease}

#     # model_path=models/${model_id}
#     model_path=isspek/${model_id}
#     train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
#     val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
#     test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

#     echo $model_id
#     echo Train disease $train_diseases
#     echo Test disease $test_diseases

#     mkdir -p $predictions_dir

#     python3 -m code.classification.transformers.model_ktrain \
#     --cuda_device $cuda_device \
#     --pretrained_model $pretrained_model \
#     --random_seed $random_seed \
#     --learning_rate $learning_rate \
#     --train_batch $train_batch \
#     --model_path $model_path \
#     --num_epochs $num_epochs \
#     --eval \
#     --train_dataset $train_dataset \
#     --test_dataset $test_dataset \
#     --val_dataset $val_dataset \
#     --predictions_dir $predictions_dir \
#     --train_diseases $train_diseases \
#     --test_disease $test_disease \
#     --model normal \
#     --model_id $model_id \
#     --train \
#     --class_weight

# echo $disease
#     echo $n_fold results:
#     python3 -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

#     done
# done
# done

# train_diseases=ebola
# test_disease=covid
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python3 -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python3 -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done

# test_disease=monkeypox
# for train_diseases in covid_zika covid_zika_ebola;
# do
# for n_fold in 1 2 3 4 5;
# do
# n_fold=$n_fold

# model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}_weight
# echo training $model_id
# predictions_dir=data/results/${model_id}/${test_disease}
# model_path=models/${model_id}
# train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
# val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

# echo $model_id
# echo Train disease $train_diseases
# echo Test disease $test_diseases

# mkdir -p $predictions_dir

# python3 -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train \
# --class_weight

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python3 -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv
# done
# done
for train_diseases in ebola;
do
for n_fold in 1 2 3 4 5;
do
n_fold=$n_fold

model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}
echo training $model_id
model_path=models/${model_id}
train_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
val_dataset=data/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
# test_dataset=data/covid/kfold/fold_${n_fold}/processed_v2/test.tsv
# test_disease=covid

echo $model_id
echo Train disease $train_diseases
# echo Test disease $test_diseases


# echo covid
# python3 -m code.classification.transformers.model_ktrain \
# --cuda_device $cuda_device \
# --pretrained_model $pretrained_model \
# --random_seed $random_seed \
# --learning_rate $learning_rate \
# --train_batch $train_batch \
# --model_path $model_path \
# --num_epochs $num_epochs \
# --eval \
# --train_dataset $train_dataset \
# --test_dataset $test_dataset \
# --val_dataset $val_dataset \
# --predictions_dir $predictions_dir \
# --train_diseases $train_diseases \
# --test_disease $test_disease \
# --model normal \
# --model_id $model_id \
# --train

# echo trained $train_diseases, tested on $test_disease
#     echo $n_fold results:
#     python3 -m code.classification.eval \
#     --labels ${test_dataset} \
#     --preds ${predictions_dir}/results.tsv

# echo monkeypox
for test_disease in zika covid monkeypox;
do
predictions_dir=data/results/${model_id}/${test_disease}
test_disease=${test_disease}
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
--model_id $model_id

echo trained $train_diseases, tested on $test_disease
    echo $n_fold results:
    python3 -m code.classification.eval \
    --labels ${test_dataset} \
    --preds ${predictions_dir}/results.tsv
done
done

done