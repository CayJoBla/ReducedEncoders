import torch
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from queue import Empty
from sentence_transformers import SentenceTransformer
import argparse
from datasets import load_dataset
import wandb

from reduced_encoders import DimReduce, DimExpand, Qwen2ReducedConfig
from reduced_encoders.modeling_utils import dimensional_distillation_loss

# # Params
# LR = 2e-4
# BATCH_SIZE = 16
# EMBEDDING_MODEL_NAME = "Alibaba-NLP/gte-Qwen1.5-7B-instruct"

def inference_worker(gpu_id, config, input_queue, embedding_queue, **kwargs):
    """Worker that runs sentence embedding model on a given GPU."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    # Load the sentence embedding model
    model = SentenceTransformer(
        model_name_or_path=config["model"], 
        device=device,
        **kwargs    # TODO: add trust_remote_code=True
    )
    
    while True:
        try:
            sentence = input_queue.get()
            if sentence is None:    # Stop signal
                break
            embedding = model.encode(sentence, convert_to_tensor=True).cpu()
            embedding_queue.put(embedding)
            # print(f"[Inference Worker {gpu_id}] Submitted embedding.")
        except Exception as e:
            print(f"[Inference Worker {gpu_id}] Error: {e}")
            break

    print(f"[Inference Worker {gpu_id}] Finished.")

def training_worker(gpu_id, config, embedding_queue, **kwargs):
    """Worker that collects embeddings into batches and trains the model."""
    torch.cuda.set_device(gpu_id)           # Set training GPU
    device = torch.device(f"cuda:{gpu_id}")

    # Initialize model to train and loss function
    model_config = Qwen2ReducedConfig.from_pretrained(config["model"], **kwargs)
    reduce = DimReduce(model_config).to(device)
    expand = DimExpand(model_config).to(device)
    alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True, device=device))
    beta = nn.Parameter(torch.tensor(0.5, requires_grad=True, device=device))
    loss_fn = dimensional_distillation_loss

    # Initialize wandb logging
    config["do_wandb_logging"] = config["run_name"] is not None
    if config["do_wandb_logging"]:
        wandb.init(
            project="reduced-encoders", 
            name=config["run_name"],
            config = {
                "learning_rate": config["lr"],
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"],
                "base_model": config["model"],
            }
        )

    # Initial evaluation
    print(f"[Training Worker {gpu_id}] Running initial evaluation...")
    run_eval(0, config, device, reduce, expand, alpha, beta, loss_fn)

    # Initialize optimizer
    params = list(reduce.parameters()) + list(expand.parameters()) + [alpha, beta]
    optimizer = torch.optim.Adam(
        params, 
        lr=config["lr"],
    )
    # TODO: Learning rate scheduler
    
    i = 0
    while True:
        # Collect embeddings until we have a full batch
        batch = []
        while len(batch) < config["batch_size"]:
            if not embedding_queue.empty():
                embedding = embedding_queue.get()
                if embedding is None:   # Stop signal
                    print(f"[Training Worker {gpu_id}] Received stop signal.")
                    return
                else:
                    batch.append(embedding)
    
        full_embeddings = torch.stack(batch).to(device)

        # print(f"[Training Worker {gpu_id}] Received batch of size {full_embeddings.shape}.")

        # Forward pass
        reduced_embeddings = reduce(full_embeddings)
        reconstructed_embeddings = expand(reduced_embeddings)
        loss, contrastive_loss, reconstruction_loss = loss_fn(
            full_embeddings, 
            reduced_embeddings, 
            reconstructed_embeddings, 
            alpha, 
            beta
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        i += 1

        # Logging
        if i % config["logging_steps"] == 0:
            print(f"[Training Worker {gpu_id}] Step {i}: Loss = {loss.item()}")
            if config["do_wandb_logging"]:
                wandb.log({
                    "train/global_step": i,
                    "train/loss": loss.item(),
                    "train/contrastive_weight": alpha.item(),
                    "train/reconstruction_weight": beta.item(),
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/reconstruction_loss": reconstruction_loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                })

        # Evaluation
        if i % config["eval_steps"] == 0:
            print(f"[Training Worker {gpu_id}] Running evaluation...")
            run_eval(i, config, device, reduce, expand, alpha, beta, loss_fn)

        # Save model
        if i % config["save_steps"] == 0:
            print(f"[Training Worker {gpu_id}] Saving model...")
            # TODO: Save model

    # Final evaluation
    print(f"[Training Worker {gpu_id}] Running final evaluation...")
    run_eval(i, config, device, reduce, expand, alpha, beta, loss_fn)

    # Save final model
    print(f"[Training Worker {gpu_id}] Saving model...")
    # TODO: Save model


def run_eval(step, config, device, reduce, expand, alpha, beta, loss_fn):
    reduce.eval()
    expand.eval()
    with torch.no_grad():
        eval_loss = 0
        eval_contrastive_loss = 0
        eval_reconstruction_loss = 0
        for batch in config["eval_dataloader"]:
            full_embeddings = batch.to(device)
            reduced_embeddings = reduce(full_embeddings)
            reconstructed_embeddings = expand(reduced_embeddings)
            loss, contrastive_loss, reconstruction_loss = loss_fn(
                full_embeddings, 
                reduced_embeddings, 
                reconstructed_embeddings, 
                alpha, 
                beta
            )
            eval_loss += loss.item()
            eval_contrastive_loss += contrastive_loss.item()
            eval_reconstruction_loss += reconstruction_loss.item()
    
        num_batches = len(config["eval_dataloader"])
        eval_loss /= num_batches
        eval_contrastive_loss /= num_batches
        eval_reconstruction_loss /= num_batches

    print(f"[Evaluation] Step: {step}, Eval Loss: {eval_loss}")
    if config["do_wandb_logging"]:
        wandb.log({
            "eval/global_step": step,
            "eval/loss": eval_loss, 
            "eval/contrastive_loss": eval_contrastive_loss, 
            "eval/reconstruction_loss": eval_reconstruction_loss
        })

    reduce.train()
    expand.train()

def scheduling_worker(config):
    # Get device counts
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        raise ValueError("This script requires at least 2 GPUs.")
    num_inference_workers = num_gpus - 1

    # Start multiprocessing
    mp.set_start_method("spawn", force=True)
    input_queue = mp.Queue()
    embedding_queue = mp.Queue()
    
    # Start inference workers
    inference_processes = []
    for i in range(num_inference_workers):
        p = mp.Process(
            target=inference_worker, 
            args=(i, config, input_queue, embedding_queue),
        )
        p.start()
        inference_processes.append(p)

    # Send eval data to inference workers
    num_eval_inputs = 0
    for batch in config.get("eval_dataloader", []):
        if isinstance(batch, dict):
            batch = batch["text"]
        for sentence in batch:
            input_queue.put(sentence)
            num_eval_inputs += 1
    print(f"Sent {num_eval_inputs} evaluation sentences to inference workers.")

    # Collect eval embeddings from inference workers
    eval_embeddings = []
    while len(eval_embeddings) < num_eval_inputs:
        if not embedding_queue.empty():
            embedding = embedding_queue.get()
            eval_embeddings.append(embedding)
    print(f"Collected {len(eval_embeddings)} eval embeddings.")

    # Create a new dataloader for eval embeddings
    eval_dataloader = DataLoader(
        eval_embeddings, 
        batch_size=config["batch_size"],
    )
    config["eval_dataloader"] = eval_dataloader

    # Start training worker
    trainer_process = mp.Process(
        target=training_worker, 
        args=(num_gpus-1, config, embedding_queue),
    )
    trainer_process.start()
    
    # Send training data to inference workers
    for epoch in range(config["num_epochs"]):
        for batch in config.get("train_dataloader", []):
            for sentence in batch:
                input_queue.put(sentence)
    
    # Send stop signals
    for _ in range(num_inference_workers):
        input_queue.put(None)
    
    # Wait for inference workers to finish
    for p in inference_processes:
        p.join()

    # Send stop signal to training worker
    embedding_queue.put(None)
    trainer_process.join()

def main(
    embedding_model=None, 
    dataset_path=None, 
    dataset_name=None, 
    split="train", 
    train_size=None, 
    validation_size=None, 
    batch_size=16,
    learning_rate=2e-4,
    num_epochs=1,
    eval_steps=50000,
    save_steps=50000, 
    logging_steps=50, 
    output_dir=None, 
    run_name=None, 
    seed=None, 
    verbose=False
):
    # TODO: Finish this function

    # Set seed
    if seed is not None:
        set_seed(seed)

    # Load and preprocess dataset
    dataset = load_dataset(dataset_path, dataset_name, split=split)  
    input_data = dataset.train_test_split(
        train_size=train_size, test_size=validation_size
    )

    # Create dataloaders
    # NOTE: May need drop_last=True
    train_dataloader = DataLoader(input_data["train"], shuffle=True, batch_size=batch_size, drop_last=True)
    eval_dataloader = DataLoader(input_data["test"], batch_size=batch_size, drop_last=True)
    # train_dataloader = DataLoader(["hello "+str(i) for i in range(160)], batch_size=batch_size)
    # eval_dataloader = DataLoader(["hello "+str(i) for i in range(32)], batch_size=batch_size)

    # Begin training
    config = {
        "model": embedding_model,
        "train_dataloader": train_dataloader,
        "eval_dataloader": eval_dataloader,
        "batch_size": batch_size,
        "lr": learning_rate,
        "num_epochs": num_epochs,
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "logging_steps": logging_steps,
        "output_dir": output_dir,
        "run_name": run_name,
    }
    scheduling_worker(config)

    # TODO: Need to figure out a way to handle the lr scheduler
    # TODO: Probably want to create the model and optimizer here, then pass them
    #       to the training process. This way, you can save the model from the 
    #       main process after training is complete.
    # TODO: The training loop is fairly implemented already, but you need to 
    #       figure out how to do logging, evaluation, and saving for this script.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a dimensionality reduction with compression")
    parser.add_argument(
        '--model',
        help=("The name or path to the base sentence embedding model."),
        type=str,
        required=True,
        dest="embedding_model"
    )
    parser.add_argument(
        '--dataset',
        '--dataset_path',
        help=("The dataset to train the model on. Required for training."),
        type=str,
        required=True,
        dest="dataset_path"
    )
    parser.add_argument(
        '--dataset_name',
        help=("The name of the data subset to use within the dataset_path dataset. Default is None."),
        type=str,
        default=None,
    )
    parser.add_argument(
        '--split',
        help=("The split of the dataset to use for training. Default split is 'train'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--train_size',
        help=("The size of the training set to use. Default is 0.9."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--validation_size',
        help=("The size of the validation set when evaluating during training. Default is 1-{train_size}."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--batch_size',
        help=("The batch size to use during training. Default is 16."),
        type=int,
        default=16
    )
    parser.add_argument(
        '--learning_rate',
        help=("The learning rate to use during training. Default is 2e-4."),
        type=float,
        default=2e-4
    )
    parser.add_argument(
        '--num_epochs',
        help=("The number of epochs to train for. Default is 1."),
        type=int,
        default=1
    )
    parser.add_argument(
        '--eval_steps',
        help=("The number of steps between evaluations during training. Default is 50000."),
        type=int,
        default=50000
    )
    parser.add_argument(
        '--save_steps',
        help=("The number of steps between saves during training. Default is 50000."),
        type=int,
        default=50000
    )
    parser.add_argument(
        '--logging_steps',
        help=("The number of steps between logging during training. Default is 50."),
        type=int,
        default=50
    )
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the trained model to. If None, the model name is used."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--run_name',
        help=("The name of the WandB run. Default is 'pretrain-compressed-{dataset_path}'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--seed',
        help=("The seed to use for randomness in preprocessing. Default is None."),
        type=int,
        default=None
    )
    parser.add_argument(
        '--verbose',
        '-v',
        help=("Whether to print preprocessing progress information. Default is False."),
        default=False,
        action="store_true"
    )  

    kwargs = parser.parse_args()
    main(**vars(kwargs))