export WANDB_PROJECT=bert-base-uncased-reduced-glue
export CUDA_VISIBLE_DEVICES="0"
export TASK_NAME=$1

python evaluate_bert.py \
  --model cayjobla/bert-base-uncased-reduced \
  --revision glue \
  --task $TASK_NAME \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 3 \
  --logging_steps 10 \
  --output_dir bert-base-uncased-reduced \
  --run_name glue-$TASK_NAME \
  -v
