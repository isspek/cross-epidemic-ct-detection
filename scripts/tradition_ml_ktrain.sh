random_seed=0

for disease in zika_ebola;
do
echo $disease training and testing
train_diseases=zika
test_disease=$disease

 for n_fold in 1 2 3 4 5;
 do
 model_id=${model_name}_${train_diseases}_${n_fold}_${learning_rate}_${train_batch}_tradition_ml
 echo training $model_id
 predictions_dir=dataset/results/${model_id}/${test_disease}
 model_path=models/${model_id}
 train_dataset=dataset/${train_diseases}/kfold/fold_${n_fold}/processed_v2/train.tsv
 val_dataset=dataset/${train_diseases}/kfold/fold_${n_fold}/processed_v2/val.tsv
 test_dataset=dataset/${test_disease}/kfold/fold_${n_fold}/processed_v2/test.tsv

 echo Train disease $train_diseases
 echo Test disease $test_diseases

 mkdir -p $predictions_dir

 python -m code.classification.tradition.model_ktrain \
 --random_seed $random_seed \
 --model_path $model_path \
 --train_dataset $train_dataset \
 --test_dataset $test_dataset \
 --val_dataset $val_dataset \
 --predictions_dir $predictions_dir \
 --train_diseases $train_diseases \
 --test_disease $test_disease \
 --train \
 --model_id $model_id \
 --eval

   echo $disease
   echo $n_fold results:
   python -m code.classification.eval \
   --labels ${test_dataset} \
   --preds ${predictions_dir}/results.tsv

 done
 done