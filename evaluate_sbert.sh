export WANDB_PROJECT=all-mpnet-base-v2-reduced-glue
export CUDA_VISIBLE_DEVICES="0,1,2"
export TASK_NAME=$1

python evaluate_mpnet_sbert.py \
  --model cayjobla/all-mpnet-base-v2-reduced \
  --tokenizer sentence-transformers/all-mpnet-base-v2 \
  --revision main \
  --task $TASK_NAME \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 3 \
  --logging_steps 20 \
  --output_dir all-mpnet-base-v2-reduced \
  --run_name glue-$TASK_NAME \
  --seed 42 \
  -v

