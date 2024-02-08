WANDB_PROJECT="mpnet-compressed" \
python train_compressed.py \
    --model cayjobla/all-mpnet-base-v2-compressed \
    --revision main \
    --dataset_path "wikipedia" \
    --dataset_name "20220301.en" \
    --split "train[:1%]" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --batch_size 16 \
    --eval_strategy steps \
    --eval_steps 2500 \
    --save_strategy steps \
    --save_steps 2500 \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --logging_steps 100 \
    --output_dir "all-mpnet-base-v2-compressed" \
    --run_name "train-wikipedia-1%" \
    -v \
    # --checkpoint "latest" 