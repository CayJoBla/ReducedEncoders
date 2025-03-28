# train_compressed_mistral_accelerate.py

import torch
from torch import nn, optim
from sentence_transformers import SentenceTransformer
from datasets import Dataset, DatasetDict
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
import os
import random
import numpy as np
import torch

from reduced_encoders.modeling_reduced_new import MultiResize
from reduced_encoders.modeling_utils import DimensionalReductionLoss

# -------------------------------- Configuration -------------------------------

SEED = 916

# Base model config
BASE_MODEL = "Linq-AI-Research/Linq-Embed-Mistral"
MAX_SEQ_LENGTH = 4096

# Reduction config
DROPOUT = 0.1
SIZES = [4096,3584,2048,1536,1024,768,512,384,256,128,64]
ACT_FN = nn.SiLU()

# Dataset config
MEDI_DATA_PATH = "/home/cayjobla/ReducedEncoders/medi-data/medi-data.json"
EVAL_SIZE = 8000    # NOTE: Change?

# Training config
RUN_NAME = "mistral-reduced"
LR = float(os.getenv("TRAIN_LR"))
BATCH_SIZE = 4
NUM_EPOCHS = 1
OUTPUT_DIR = "/home/cayjobla/ReducedEncoders/Linq-Embed-Mistral-reduced"
LOGGING_STEPS = 2500
CLEAR_CUDA_CACHE_STEPS = 100
EVAL_STEPS = 50000
SAVE_STEPS = 50000

# ------------------------------ Initialization --------------------------------

# Set seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)

# Initialize the accelerator
accelerator = Accelerator(log_with="wandb")

# ---------------------------- Model Initialization ----------------------------

# Load the model
with accelerator.main_process_first():  # Allow secondary processes to load from cache
    model = SentenceTransformer(BASE_MODEL, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH

# Prepare the model for inference (no training on full sentence transformer)
for param in model.parameters():
    param.requires_grad = False
model.eval()

accelerator.wait_for_everyone()
accelerator.print(f"Loaded model: {BASE_MODEL}")

# Initialize the reduction and reconstruction layers
reduce = MultiResize(
    sizes=SIZES, 
    bias=True, 
    activation_function=ACT_FN,
    dropout=DROPOUT
).to("cpu")
expand = MultiResize(
    sizes=SIZES[::-1], 
    bias=True, 
    activation_function=ACT_FN,
    dropout=DROPOUT
).to("cpu")

accelerator.wait_for_everyone()
accelerator.print(f"Initialized reduction and expansion layers with sizes: {SIZES}")

# --------------------------- Dataset Initialization ---------------------------

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def load_medi_dataset(data_filepath):
    with open(data_filepath, "r") as f:
        json_data = json.load(f)
    return Dataset.from_dict({
        "query": [get_detailed_instruct(*item["query"]) for item in json_data],
        "pos": [item["pos"][1] for item in json_data],  # No instruction for documents
        "neg": [item["neg"][1] for item in json_data],
        "task_name": [item["task_name"] for item in json_data]
    })

# Load the dataset
dataset = load_medi_dataset(MEDI_DATA_PATH)

# Split the dataset
train_indices, eval_indices = train_test_split(
    list(range(len(dataset))),      # Indices of dataset
    test_size=EVAL_SIZE,
    stratify=dataset["task_name"],  # Ensures proportional split of tasks
)
dataset = DatasetDict({
    "train": dataset.select(train_indices),
    "eval": dataset.select(eval_indices)
})

# Initialize the dataloaders
train_dataloader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
eval_dataloader = DataLoader(dataset["eval"], batch_size=BATCH_SIZE, shuffle=False)

# TESTING: Downsample the dataset
# train_dataloader = DataLoader(dataset["train"].select(list(range(0, 1000))), batch_size=BATCH_SIZE, shuffle=True)
# eval_dataloader = DataLoader(dataset["eval"].select(list(range(0, 30))), batch_size=BATCH_SIZE, shuffle=False)

accelerator.wait_for_everyone()
accelerator.print(f"Loaded medi dataset: {MEDI_DATA_PATH}, eval size: {EVAL_SIZE}")

# --------------------------- Loss Initialization ------------------------------

# Initialize the loss function
loss_fn = DimensionalReductionLoss()
accelerator.wait_for_everyone()
accelerator.print(f"Initialized DimensionalReductionLoss with weights: {loss_fn.weights.detach()}")

# ------------------------- Optimizer Initialization ---------------------------

# Initialize the optimizer
trainable_params = list(reduce.parameters()) + list(expand.parameters()) + list(loss_fn.parameters())
optimizer = optim.AdamW(trainable_params, lr=LR)

# Initialize lr scheduler
num_training_steps = len(train_dataloader) * NUM_EPOCHS
lambda_lr = lambda step: max(0, (num_training_steps - step) / num_training_steps)
lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)

accelerator.wait_for_everyone()
accelerator.print(f"Initialized optimizer and LR scheduler")

# ---------------------------- Training Prep -----------------------------------

def forward(batch):
    with torch.no_grad():
        full_embeddings = model.encode(
            batch["query"] + batch["pos"] + batch["neg"],
            convert_to_tensor=True,
            show_progress_bar=False
        ).detach()

    reduced_embeddings = reduce(full_embeddings)
    reconstructed_embeddings = expand(reduced_embeddings)

    losses = loss_fn(reduced_embeddings, reconstructed_embeddings, full_embeddings)
    del full_embeddings, reduced_embeddings, reconstructed_embeddings

    return losses

def log():
    epoch_frac = global_step / len(train_dataloader)
    accelerator.print(f"Epoch: {epoch_frac}, Step: {global_step}, Train Loss: {running_loss / running_count}")
    cosine_sim_weight, reconstruction_weight = accelerator.unwrap_model(loss_fn).weights.detach()
    accelerator.log({
        "train/epoch": epoch_frac,
        "train/global_step": global_step,
        "train/loss": running_loss / running_count,
        "train/cosine_sim_loss": running_l_c / running_count,
        "train/reconstruction_loss": running_l_r / running_count,
        "train/cosine_sim_weight": cosine_sim_weight.item(),
        "train/reconstruction_weight": reconstruction_weight.item(),
        "train/learning_rate": optimizer.param_groups[0]["lr"],
    })

def evaluate():
    accelerator.print(f"Running evaluation (step {global_step})...")
    reduce.eval()
    expand.eval()
    loss_fn.eval()

    eval_loss = []
    eval_l_c = []
    eval_l_r = []
    for eval_batch in eval_dataloader:
        with torch.no_grad():
            loss, l_c, l_r = forward(eval_batch)
            eval_loss.append(loss.detach())
            eval_l_c.append(l_c.detach())
            eval_l_r.append(l_r.detach())

    eval_loss = accelerator.gather(torch.stack(eval_loss)).mean().item()
    eval_l_c = accelerator.gather(torch.stack(eval_l_c)).mean().item()
    eval_l_r = accelerator.gather(torch.stack(eval_l_r)).mean().item()

    epoch_frac = global_step / len(train_dataloader)
    accelerator.print(f"Epoch: {epoch_frac}, Step: {global_step}, Eval Loss: {eval_loss}")
    accelerator.log({
        "train/epoch": epoch_frac,
        "train/global_step": global_step,
        "eval/loss": eval_loss,
        "eval/cosine_sim_loss": eval_l_c,
        "eval/reconstruction_loss": eval_l_r,
    })

    reduce.train()
    expand.train()
    loss_fn.train()

def save(checkpoint=None):
    output_dir = OUTPUT_DIR
    if checkpoint is not None:
        output_dir = os.path.join(output_dir, f"checkpoint-{checkpoint}")
    os.makedirs(output_dir, exist_ok=True)

    accelerator.wait_for_everyone()
    reduce_unwrapped = accelerator.unwrap_model(reduce)
    accelerator.save(reduce_unwrapped.state_dict(), os.path.join(output_dir, "reduce.pth"))
    expand_unwrapped = accelerator.unwrap_model(expand)
    accelerator.save(expand_unwrapped.state_dict(), os.path.join(output_dir, "expand.pth"))
    loss_fn_unwrapped = accelerator.unwrap_model(loss_fn)
    accelerator.save(loss_fn_unwrapped.state_dict(), os.path.join(output_dir, "loss_fn.pth"))

# Prepare model with accelerator
model, reduce, expand, loss_fn, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
    model, reduce, expand, loss_fn, optimizer, lr_scheduler, train_dataloader, eval_dataloader
)
accelerator.init_trackers(
    project_name="reduced-encoders",
    config={
        "dropout": DROPOUT, 
        "sizes": SIZES, 
        "activation_function": ACT_FN.__class__.__name__,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
    },
    init_kwargs={"wandb": {"name": RUN_NAME}}
)
accelerator.wait_for_everyone()
accelerator.print(f"Initialized wandb tracking with run name: {RUN_NAME}")

# ---------------------------- Training Loop -----------------------------------

reduce.train()
expand.train()
loss_fn.train()

accelerator.wait_for_everyone()
accelerator.print("Starting training loop...")

running_loss = 0
running_l_c = 0
running_l_r = 0
running_count = 0

global_step = 0
for epoch in range(NUM_EPOCHS):
    # On epoch start
    evaluate()
    save(f"epoch-{epoch}")

    for batch in train_dataloader:
        # Forward pass
        loss, l_c, l_r = forward(batch)

        # Backward pass
        optimizer.zero_grad()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()    # Update learning rate

        # Log running loss
        running_loss += loss.item()
        running_l_c += l_c.item()
        running_l_r += l_r.item()
        running_count += 1

        # Clear cuda cache
        if global_step % CLEAR_CUDA_CACHE_STEPS == 0:
            torch.cuda.empty_cache()

        # Training logs
        if global_step % LOGGING_STEPS == 0:
            log()

        global_step += 1    # Avoid rerunning eval/save on first step

        # Evaluation
        if global_step % EVAL_STEPS == 0:
            accelerator.wait_for_everyone()
            evaluate()

        # Save
        if global_step % SAVE_STEPS == 0:
            accelerator.wait_for_everyone()
            save()

# Final evaluation and save
accelerator.wait_for_everyone()
evaluate()
save()
