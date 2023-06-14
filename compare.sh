python train_bert.py --model="bert-base-uncased" --dataset="cayjobla/wikipedia_tokenized" \
    --num_shards=20 --index=$1 --batch_size=128 --finetune_base=False --num_epochs=1 --scheduler_type="linear" \
    --num_warmup_steps=0 --logging_strat="step" --logging_steps=3000 --push_to_hub=False \
    --wandb_project="bert-base-uncased-reduced" --run_name="bert-s$1"