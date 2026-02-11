export CUDA_VISIBLE_DEVICES=4


#for data_id in FD001 FD002 FD003 FD004
#do
#
#for M_name in TwoP_Transformer
#do
#
#for loss in MAE
#do
#
#for length in  40 50 60
#do
#
#for rate in  0.0001 0.00005 0.00001 0.000001
#do
#
#python -u main.py \
#  --dataset_name 'CMAPSS'\
#  --Data_id_CMAPSS $data_id\
#  --train True\
#  --resume False\
#  --input_length $length\
#  --batch_size 64\
#  --d_model 64\
#  --d_ff 128\
#  --dropout 0.1\
#  --model_name $M_name\
#  --info 'TwoP_Transformer baseline '\
#  --loss_type $loss\
#  --train_epochs 150\
#  --learning_rate $rate\
#  --is_minmax True\
#
#done
#
#done
#
#done
#
#done
#
#done
#
#
#
#
#
#for data_id in DS01 DS03 DS05 DS07
#do
#
#for M_name in TwoP_Transformer
#do
#
#for loss in MAE
#do
#
#for length in  40 50 60
#do
#
#for rate in   0.0001 0.00005 0.00001 0.000001
#do
#
#python -u main.py \
#  --dataset_name 'N_CMAPSS'\
#  --Data_id_N_CMAPSS $data_id\
#  --train True\
#  --resume False\
#  --s 10\
#  --sampling 50\
#  --change_len True\
#  --rate 0.8\
#  --input_length $length\
#  --batch_size 64\
#  --d_model 64\
#  --d_ff 128\
#  --dropout 0.1\
#  --model_name $M_name\
#  --info 'TwoP_Transformer baseline'\
#  --loss_type $loss\
#  --train_epochs 150\
#  --learning_rate $rate\
#  --is_minmax True\
#
#done
#
#done
#
#done
#
#done
#
#done


for data_id in 35 37 40
do

for M_name in TwoP_Transformer
do

for loss in MAE
do

for length in  50 60 80
do

for rate in   0.00005 0.00001 0.000001
do

python -u main.py \
  --dataset_name 'XJTU'\
  --Data_id_XJTU $data_id\
  --DA False\
  --Classify False\
  --train True\
  --resume False\
  --s 10\
  --sampling 10\
  --rate 0.8\
  --xjtu_n_fea 40\
  --input_length $length\
  --batch_size 64\
  --d_model 64\
  --d_ff 128\
  --dropout 0.1\
  --model_name $M_name\
  --info 'XJTU baseline n_feature 40 sample 10 s 10'\
  --loss_type $loss\
  --train_epochs 50\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

done

done
