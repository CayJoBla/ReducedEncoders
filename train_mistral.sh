export CUDA_VISIBLE_DEVICES="0,1,2"
export TRAIN_LR=0.0001
accelerate launch --multi_gpu train_compressed_mistral_accelerate.py