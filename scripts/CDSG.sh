export CUDA_VISIBLE_DEVICES=1


#for data_id in FD001 FD002 FD003 FD004
#do
#
#for M_name in CDSG
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
#  --info '11.5 CDSG gru ablation '\
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





for data_id in DS01 DS03 DS05 DS07
do

for M_name in CDSG
do

for loss in MAE
do

for length in  40 50 60
do

for rate in   0.0001 0.00005 0.00001 0.000001
do

python -u main.py \
  --task 'normal'\
  --dataset_name 'N_CMAPSS'\
  --Data_id_N_CMAPSS $data_id\
  --train True\
  --resume False\
  --s 10\
  --sampling 50\
  --change_len True\
  --rate 0.8\
  --input_length $length\
  --batch_size 64\
  --d_model 64\
  --d_ff 128\
  --dropout 0.1\
  --model_name $M_name\
  --info 'CDSG baseline'\
  --loss_type $loss\
  --train_epochs 150\
  --learning_rate $rate\
  --is_minmax True\

done

done

done

done

done
