#!/bin/bash

#SBATCH --job-name=TEST_VQVAE3_ETTh1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4 
#SBATCH --partition=P2
#SBATCH --output=/home/s2/jinmyeongchoi/nhits-tsl/TEST_VQVAE_log/S-%x.%j.out


model_name=TEST_VQVAE3
dataset=ETTh1

pred_len=(96 192 336 720)
for pred_len in ${pred_len[@]}
do
  python -u /home/s2/jinmyeongchoi/nhits-tsl/run_vqvae.py \
    --is_training 1 \
    --seed 7 \
    --batch_size 128 \
    --train_epochs 10 \
    --task_name long_term_forecast \
    --model_id ${model_name}_${pred_len} \
    --model ${model_name} \
    --data ${dataset} \
    --root_path /shared/s2/lab01/timeSeries/forecasting/base/ETT-small/ \
    --checkpoints /shared/s2/lab01/jinmyeongchoi/TEST_VQVAE_checkpoints/ \
    --data_path ETTh1.csv \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len ${pred_len} \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --d_ff 2048 \
    --patch_size 16 \
    --num_embeddings 128 \
    --poly_degree 2 \
    --num_harmonics 2 \
    --des 'test_test_vqvae3'
done
