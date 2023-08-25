WANDB_PROJECT="bert-base-uncased-reduced-pretrain" \
python pretrain_bert.py \
    --model cayjobla/bert-base-uncased-reduced \
    --revision pretrain \
    --dataset "cayjobla/wikipedia-pretrain-processed" \
    --split "train" \
    --train_size 0.99 \
    --validation_size 0.01 \
    --mlm_probability 0.15 \
    --batch_size 16 \
    --eval_strategy steps \
    --eval_steps 25000 \
    --save_strategy steps \
    --save_steps 25000 \
    --learning_rate 2e-4 \
    --num_epochs 1 \
    --logging_steps 50 \
    --output_dir "bert-base-uncased-reduced" \
    --run_name "mlm-nsp-wikipedia" \
    -v \
    # --checkpoint "latest" 