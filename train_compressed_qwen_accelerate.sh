WANDB_PROJECT="qwen2-compressed" \
# CUDA_VISIBLE_DEVICES="0,1" \
python train_compressed_qwen_accelerate.py \
    --model cayjobla/gte-Qwen1.5-7B-instruct-reduced \
    --revision main \
    --dataset_path "wikipedia" \
    --dataset_name "20220301.en" \
    --split "train[:100]" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --batch_size 16 \
    --eval_steps 25000 \
    --save_steps 25000 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --logging_steps 2500 \
    --output_dir "gte-Qwen1.5-7B-instruct-reduced" \
    --run_name "train-wikipedia" \
    -v \