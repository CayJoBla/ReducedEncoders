WANDB_PROJECT="mpnet-compressed-hyperopt" \
python train_hyperopt_compressed.py \
    --model cayjobla/all-mpnet-base-v2-compressed \
    --revision main \
    --dataset_path "wikipedia" \
    --dataset_name "20220301.en" \
    --split "train[:1%]" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --num_epochs 1 \
    --logging_steps 100 \
    --output_dir "all-mpnet-base-v2-compressed" \
    --run_name "hyperopt-wikipedia-5%" \
    -v \