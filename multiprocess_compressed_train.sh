#!/bin/bash

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=65536M   # memory per CPU core
#SBATCH -J "qwen2-reduced-pretrain"   # job name
#SBATCH --mail-user=cayjobla@byu.edu   # email address
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

export WANDB_MODE="disabled"
python multiprocess_compressed_train.py \
    --model "Alibaba-NLP/gte-Qwen1.5-7B-instruct" \
    --dataset "wikipedia" \
    --dataset_name "20220301.en" \
    --split "train[:160]" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --batch_size 16 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --eval_steps 25 \
    --save_steps 25000 \
    --logging_steps 5 \
    --output_dir "gte-Qwen1.5-7B-instruct-reduced" \
    --run_name "qwen1.5-compressed-pretrain" \
    -v \
    