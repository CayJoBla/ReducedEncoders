export WANDB_PROJECT=bert-base-uncased-reduced-mlm

python run_mlm.py \
  --model_name_or_path cayjobla/bert-base-uncased-reduced \
  --model_revision initial \
  --dataset_name wikipedia \
  --dataset_config_name "20220301.en" \
  --do_train \
  --do_eval \
  --max_seq_length 256 \
  --per_device_train_batch_size 16 \
  --evaluation_strategy steps \
  --eval_steps 50000 \
  --per_device_eval_batch_size 16 \
  --save_strategy steps \
  --save_steps 50000 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --output_dir bert-base-uncased-reduced \
  --logging_steps 50 \
  --run_name mlm-wikipedia \
  --overwrite_output_dir \
  # --resume_from_checkpoint bert-base-uncased-reduced/checkpoint-200000   # Only for fixing checkpoint

