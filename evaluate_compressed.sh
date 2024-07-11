export WANDB_PROJECT=mpnet-compressed-glue
export CUDA_VISIBLE_DEVICES="1"
export TASK_NAME=$1

python evaluate_compressed.py \
  --model cayjobla/all-mpnet-base-v2-compressed \
  --tokenizer cayjobla/all-mpnet-base-v2-compressed \
  --revision unnormalized \
  --task $TASK_NAME \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 3 \
  --logging_steps 20 \
  --output_dir all-mpnet-base-v2-compressed \
  --run_name variable-glue-$TASK_NAME \
  --seed 916 \
  -v

