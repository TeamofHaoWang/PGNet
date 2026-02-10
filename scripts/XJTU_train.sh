export CUDA_VISIBLE_DEVICES=5


for data_id in 40
do

for M_name in Diff_Degra_index Diff_index Diff_Graph_Net_1
do

for loss in MAE QUAN
do

for length in  50 60 70 80
do

for rate in   0.0001 0.00005 0.00001 0.000001
do

python -u main.py \
  --dataset_name 'XJTU'\
  --Data_id_XJTU $data_id\
  --DA False\
  --Classify False\
  --train True\
  --resume False\
  --s 10\
  --sampling 50\
  --rate 0.8\
  --xjtu_n_fea 40\
  --input_length $length\
  --batch_size 100\
  --d_model 64\
  --d_ff 128\
  --dropout 0.1\
  --model_name $M_name\
  --info 'XJTU n_feature 40 sample 50 s 10'\
  --loss_type $loss\
  --train_epochs 50\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

done

done



