export WANDB_PROJECT=reduced-encoders-glue
export CUDA_VISIBLE_DEVICES="0,1,2"
export MODEL_HF_PATH=$1
export TASK_NAME=$2

python evaluate_glue.py \
  --model $MODEL_HF_PATH \
  --task $TASK_NAME \
  --revision main \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --num_epochs 3 \
  --logging_steps 50 \
  --seed 916 \
  --do_predict \
  -v

