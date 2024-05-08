CUDA_VISIBLE_DEVICES="0" \
python train_umap.py \
    --model cayjobla/all-mpnet-base-v2-compressed \
    --revision initial \
    --dataset_path "cayjobla/wikipedia_embedded" \
    --split "train" \
    --train_size 0.9 \
    --validation_size 0.1 \
    --batch_size 32 \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --n_neighbors 10 \
    --min_dist 0.1 \
    --metric "euclidean" \
    --output_dir "all-mpnet-base-v2-compressed-umap" \
    --no_encoding \
    -v \