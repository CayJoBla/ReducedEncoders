python preprocess_data.py \
    --dataset "wikipedia" \
    --split "20220301.en" \
    --tokenizer "bert-base-uncased" \
    --revision "main" \
    --num_shards 30 \
    --train_size 0.9 \
    --test_size 0.1 \
    --output_dir "wikipedia-pretrain-processed" \
    -v 
