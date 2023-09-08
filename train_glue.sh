export WANDB_PROJECT=bert-base-uncased-reduced-glue
export TASK_NAME=$1

python run_glue.py \
  --model_name_or_path bert-base-uncased \
  --model_revision main \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy epoch \
  --per_device_eval_batch_size 16 \
  --learning_rate 2e-4 \
  --num_train_epochs 10 \
  --output_dir bert-base-uncased-reduced \
  --overwrite_output_dir \
  --logging_steps 10 \
  --run_name glue-$TASK_NAME \

python push_model.py \
  --local_dir bert-base-uncased-reduced \
  --branch glue \
  --verify \
  --commit_message "Finetune model on the $TASK_NAME dataset"


