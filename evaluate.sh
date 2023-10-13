export WANDB_PROJECT=bert-base-uncased-reduced-glue
export CUDA_VISIBLE_DEVICES="0"
export TASK_NAME=$1

python evaluate_bert.py \
  --model cayjobla/bert-base-uncased-reduced \
  --revision main \
  --task $TASK_NAME \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 3 \
  --logging_steps 50\
  --output_dir bert-base-uncased-reduced \
  --run_name glue-$TASK_NAME \
  -v

# Switch instead to fine-tuning each task individually
# Tune hyperparameters, etc for each task
# Create a new script or update script to write predictions on test set to file
# Write a script to compare prediction files to each other
# Look at predictions and scores for BERT base model and compare to reduced model

# Look at SuperGLUE tasks
