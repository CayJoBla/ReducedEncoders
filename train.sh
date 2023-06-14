python train_bert.py --model="cayjobla/bert-base-uncased-reduced" --dataset="cayjobla/wikipedia_tokenized" \
    --num_shards=20 --index=$1 --batch_size=128 --finetune_base=False --num_epochs=1 --scheduler_type="linear" \
    --num_warmup_steps=0 --logging_strat="step" --logging_steps=3000 --push_to_hub=True \
    --wandb_project="bert-base-uncased-reduced" --run_name="reduced-s$1"