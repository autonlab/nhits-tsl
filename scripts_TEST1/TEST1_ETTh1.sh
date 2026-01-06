#!/bin/bash

#SBATCH --job-name=TEST1_ETTh1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4 
#SBATCH --partition=P2
#SBATCH --output=/home/s2/jinmyeongchoi/nhits-tsl/TEST1_log/S-%x.%j.out


model_name=TEST1
dataset=ETTh1


  python -u /home/s2/jinmyeongchoi/nhits-tsl/run.py \
    --is_training 1 \
    --seed 7 \
    --batch_size 256 \
    --train_epochs 10 \
    --task_name long_term_forecast \
    --model_id ${model_name} \
    --model ${model_name} \
    --data ${dataset} \
    --root_path /shared/s2/lab01/timeSeries/forecasting/base/ETT-small/ \
    --checkpoints /shared/s2/lab01/jinmyeongchoi/TEST1_checkpoints/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers 2 \
    --d_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --d_ff 512 \
    --dropout 0.3 \
    --des 'test_test1'
