python preprocess_data.py --dataset="wikipedia" --split="20220301.en" --tokenizer="bert-base-uncased" \
    --chunk_size=128 --mlm_probability=0.15 --repo_name="wikipedia-tokenized"