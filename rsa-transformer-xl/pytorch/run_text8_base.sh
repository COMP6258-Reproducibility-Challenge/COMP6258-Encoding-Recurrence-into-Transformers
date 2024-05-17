#!/bin/bash -l
#SBATCH -p lyceum
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH -c 8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rc3g20@soton.ac.uk
#SBATCH --time=08:00:00


module load conda/py3-latest
conda activate transformer-xl



python -u train.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --n_layer 14 \
        --d_model 512 \
        --n_head 8 \
        --d_head 64 \
        --d_inner 2048 \
        --dropout 0.1 \
        --dropatt 0.0 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 0 \
        --max_step 40000 \
        --tgt_len 512 \
        --mem_len 512 \
        --eval_tgt_len 128 \
        --batch_size 22 \
        --multi_gpu \
        --gpu0_bsz 4 \
        ${@:2}


python -u eval.py \
        --cuda \
        --data ../data/text8/ \
        --dataset text8 \
        --tgt_len 80 \
        --mem_len 2100 \
        --clamp_len 820 \
        --same_length \
        --split test \
        ${@:2}
