export CUDA_VISIBLE_DEVICES=6



for data_id in FD001 FD002 FD003 FD004 
do

for M_name in  HyperNet_V2
do

for loss in QUAN
do

for length in 50 60 70 80
do

for rate in 0.0001 0.00005 0.00001 0.000001 
do

for d_model in 64
do

for d_ff in 128
do

for kernel_size in 3
do

python -u main.py \
  --task 'Hyper'\
  --dataset_name 'CMAPSS'\
  --Data_id_CMAPSS $data_id\
  --train True\
  --resume False\
  --input_length $length\
  --batch_size 64\
  --d_model $d_model\
  --d_ff $d_ff\
  --kernel_size $kernel_size\
  --dropout 0.1\
  --model_name $M_name\
  --info '11.12 gru + wo constrain loss '\
  --loss_type $loss\
  --train_epochs 150\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

done

done

done

done

done

