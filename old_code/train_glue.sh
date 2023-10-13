export WANDB_PROJECT=bert-base-uncased-reduced-glue
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=$1

python run_glue.py \
  --model_name_or_path cayjobla/bert-base-uncased-reduced \
  --model_revision main \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy epoch \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir bert-base-uncased-reduced \
  --overwrite_output_dir \
  --logging_strategy epoch \
  --save_strategy no \
  --run_name glue-$TASK_NAME \


