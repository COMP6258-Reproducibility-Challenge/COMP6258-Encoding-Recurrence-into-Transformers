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
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --n_layer 24 \
        --d_model 1024 \
        --n_head 8 \
        --d_head 128 \
        --d_inner 3072 \
        --dropout 0.15 \
        --dropatt 0.15 \
        --optim adam \
        --lr 0.00025 \
        --warmup_step 4000 \
        --max_step 400000 \
        --tgt_len 768 \
        --mem_len 768 \
        --eval_tgt_len 128 \
        --batch_size 64 \
        --multi_gpu \
        --gpu0_bsz 0 \
        ${@:2}


python -u eval.py \
        --cuda \
        --data ../data/enwik8/ \
        --dataset enwik8 \
        --tgt_len 128 \
        --mem_len 3800 \
        --clamp_len 1000 \
        --same_length \
        --split test \
        ${@:2}


f
