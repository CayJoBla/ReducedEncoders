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
    