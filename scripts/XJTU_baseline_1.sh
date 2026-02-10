export CUDA_VISIBLE_DEVICES=4

for data_id in 35 37 40
do

for M_name in SDAGCN
do

for loss in MAE
do

for length in  50 60 80
do

for rate in   0.00005 0.00001 0.000001
do

python -u main.py \
  --task 'normal'\
  --dataset_name 'XJTU'\
  --Data_id_XJTU $data_id\
  --train True\
  --resume False\
  --s 10\
  --sampling 10\
  --rate 0.8\
  --xjtu_n_fea 40\
  --input_length $length\
  --batch_size 4\
  --d_model 64\
  --d_ff 128\
  --dropout 0.1\
  --model_name $M_name\
  --info 'XJTU baseline n_feature 40 sample 10 s 10'\
  --loss_type $loss\
  --train_epochs 100\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

done

done

