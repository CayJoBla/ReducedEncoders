#!/bin/bash

#SBATCH --time=72:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=65536M   # memory per CPU core
#SBATCH -J "qwen2-reduced-pretrain"   # job name
#SBATCH --mail-user=cayjobla@byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export WANDB_MODE="offline"
export WANDB_PROJECT="qwen2-compressed"
# CUDA_VISIBLE_DEVICES="0" \
python train_compressed_qwen.py \
    --model cayjobla/gte-Qwen1.5-7B-instruct-reduced \
    --revision main \
    --dataset_path "wikipedia" \
    --dataset_name "20220301.en" \
    --split "train" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --batch_size 16 \
    --eval_strategy steps \
    --eval_steps 25000 \
    --save_strategy steps \
    --save_steps 25000 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --logging_steps 2500 \
    --output_dir "gte-Qwen1.5-7B-instruct-reduced" \
    --run_name "train-wikipedia" \
    --trust_remote_code \
    --disable_tqdm \
    -v \
    # --checkpoint "latest" 