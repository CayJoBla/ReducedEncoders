WANDB_PROJECT="mpnet-compressed" \
CUDA_VISIBLE_DEVICES="1" \
python train_compressed.py \
    --model cayjobla/all-mpnet-base-v2-compressed \
    --revision variable \
    --dataset_path "wikipedia" \
    --dataset_name "20220301.en" \
    --split "train" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --batch_size 32 \
    --eval_strategy steps \
    --eval_steps 25000 \
    --save_strategy steps \
    --save_steps 25000 \
    --learning_rate 2e-4 \
    --num_epochs 3 \
    --logging_steps 2500 \
    --output_dir "all-mpnet-base-v2-compressed" \
    --run_name "train-wikipedia-variable-no-norm" \
    --disable_tqdm \
    -v \