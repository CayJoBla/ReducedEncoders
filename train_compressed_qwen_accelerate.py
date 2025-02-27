# train_compressed.py

from transformers import set_seed, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
from reduced_encoders import Qwen2CompressedForPretraining
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import argparse
from functools import partial
from accelerate import Accelerator, infer_auto_device_map, dispatch_model
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from torch.optim.lr_scheduler import LambdaLR
import os

from reduced_encoders.modeling_utils import dimensional_distillation_loss

def train(
    model_name=None, 
    dataset_path=None, 
    dataset_name=None, 
    split="train", 
    tokenizer=None, 
    revision="main", 
    train_size=None, 
    validation_size=None, 
    batch_size=16, 
    eval_steps=50000, 
    save_steps=50000, 
    logging_steps=50, 
    learning_rate=2e-4, 
    num_epochs=1, 
    output_dir=None, 
    run_name=None, 
    seed=None, 
    verbose=False
):
    # Assumed default values (these are not passed anywhere, the script is just implemented with these parameters)
    eval_strategy = "steps"
    save_strategy = "steps"
    do_contrast = True 
    do_reconstruction = True
    checkpoint = None
    push_to_hub = False
    disable_tqdm = True

    ## Set seed
    if seed is not None:
        set_seed(seed)

    ## Load model
    if model_name is None:
        raise ValueError("The 'model' argument must be provided.")
    if verbose: 
        print(f"Loading {model_name} model...")
    model = Qwen2CompressedForPretraining.from_pretrained(
                model_name, 
                revision=revision,
                do_contrast=do_contrast,
                do_reconstruction=do_reconstruction,
            )

    ## Load tokenizer
    if tokenizer is None: 
        tokenizer = model_name
    if verbose: 
        print(f"Loading {tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision)

    ## Load data for preprocessing
    if dataset_path is None: 
        raise ValueError("The 'dataset_path' argument must be provided.")
    full_dataset_path = f"{dataset_path}:{dataset_name}" if dataset_name else dataset_path

    if verbose: print(f"Loading the '{split}' split of the '{full_dataset_path}' dataset...")
    dataset = load_dataset(dataset_path, dataset_name, split=split)

    ## Preprocess the data
    if disable_tqdm:
        disable_progress_bar()

    def preprocess_data(batch):
        return tokenizer(batch["text"], truncation=True)    # No padding, we pad in the data collator

    if verbose: 
        print(f"Preprocessing the '{split}' split of the '{full_dataset_path}' dataset...")
    preprocessed = dataset.map(preprocess_data, batched=True, batch_size=1000, remove_columns=dataset.column_names)

    ## Create a train and validation split of the preprocessed data    
    if verbose: 
        print(f"Creating a train and validation split of the preprocessed data...")
    if train_size is None and validation_size is None: # train_test_split function should handle all other cases 
        train_size = 0.9 
    input_data = preprocessed.train_test_split(train_size=train_size, test_size=validation_size, seed=seed)

    ## Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## Create DataLoader
    train_dataloader = DataLoader(input_data["train"], shuffle=True, batch_size=batch_size, collate_fn=data_collator)
    eval_dataloader = DataLoader(input_data["test"], batch_size=batch_size, collate_fn=data_collator)

    ## Extract model elements
    base_model = model.base_model
    pooler = model.pooler
    reduce_module = model.reduce
    expand_module = model.expand
    weights = model.params

    ## Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False

    ## Prepare optimizer
    trainable_params = list(reduce_module.parameters()) + list(expand_module.parameters())
    trainable_params.append(weights["contrastive_weight"])
    trainable_params.append(weights["reconstruction_weight"])
    optimizer = AdamW(trainable_params, lr=learning_rate)

    ## Learning rate scheduler
    num_training_steps = len(train_dataloader) * num_epochs
    lr_lambda = lambda step: max(0, (num_training_steps - step) / num_training_steps)
    scheduler = LambdaLR(optimizer, lr_lambda)

    ## Initialize accelerator
    accelerator = Accelerator()
    # accelerator = Accelerator(log_with="wandb")   # TODO: Add WandB logging back after debugging
    # accelerator.init_trackers(
    #     project_name=os.getenv("WANDB_PROJECT", "reduced-encoders"),
    #     config=model.config.to_dict(),
    #     # run_name=run_name,  # This doesn't work
    # )
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    ## Device placement
    max_memory = {i: "15GiB" for i in range(torch.cuda.device_count()-1)}
    train_device = torch.device("cuda", torch.cuda.device_count()-1)
    device_map = infer_auto_device_map(base_model, max_memory=max_memory)
    device_map = {"qwen2."+str(k): v for k, v in device_map.items()}
    device_map.update({
        "reduce": train_device,
        "expand": train_device,
        "params": train_device,
    })
    model = dispatch_model(model, device_map=device_map)
    
    ## Define training arguments
    if output_dir is None: 
        output_dir = model_name.split("/")[-1]
    # if run_name is None: 
    #     run_name = "pretrain-compressed-qwen2-" + dataset_path

    def forward(batch):
        # Base model inference and sentence pooling (Device 0)
        # batch = batch.to("cuda:0")
        with torch.no_grad():
            print(batch)
            outputs = base_model(**batch)   # Inference on base model
            pooled_output = pooler(outputs[0], batch["attention_mask"])

        # Reduce and reconstruct embeddings (Device 1)
        full_embeddings = pooled_output.to(train_device)
        reduced_embeddings = reduce_module(full_embeddings)
        reconstructed_embeddings = expand_module(reduced_embeddings)

        # Compute loss (constrast + reconstruction)
        return dimensional_distillation_loss(
            full_embeddings, 
            reduced_embeddings, 
            reconstructed_embeddings,
            weights["contrastive_weight"],
            weights["reconstruction_weight"],
        )

    ## Training loop
    model.train()
    global_step = 0
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            print("Step:", global_step)
            # Forward pass
            loss, contrastive_loss, reconstruction_loss = forward(batch)

            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()    # Update learning rate
            global_step += 1

            # Training logs
            if global_step % logging_steps == 0:
                if verbose:
                    print(f"Epoch: {epoch}, Step: {global_step}, Train Loss: {loss.item()}")
                accelerator.log({
                    "train/epoch": epoch,
                    "train/global_step": global_step,
                    "train/loss": loss.item(),
                    "train/contrastive_weight": alpha,
                    "train/reconstruction_weight": beta,
                    "train/contrastive_loss": contrastive_loss.item(),
                    "train/reconstruction_loss": reconstruction_loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                })

            # Evaluation
            if global_step % eval_steps == 0:
                model.eval()
                eval_loss = 0
                eval_contrastive_loss = 0
                eval_reconstruction_loss = 0
                for eval_batch in eval_dataloader:
                    with torch.no_grad():
                        loss, contrastive_loss, reconstruction_loss = forward(eval_batch)
                        eval_loss += loss.item()
                        eval_contrastive_loss += contrastive_loss.item()
                        eval_reconstruction_loss += reconstruction_loss.item()

                eval_loss /= len(eval_dataloader)
                eval_contrastive_loss /= len(eval_dataloader)
                eval_reconstruction_loss /= len(eval_dataloader)

                if verbose:
                    print(f"Epoch: {epoch}, Step: {global_step}, Eval Loss: {eval_loss}")
                accelerator.log({
                    "eval/loss": eval_loss,
                    "eval/contrastive_loss": eval_contrastive_loss,
                    "eval/reconstruction_loss": eval_reconstruction_loss,
                })
                model.train()

            if global_step % save_steps == 0:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    ## Final evaluation
    model.eval()
    eval_loss = 0
    eval_contrastive_loss = 0
    eval_reconstruction_loss = 0
    for eval_batch in eval_dataloader:
        with torch.no_grad():
            loss, contrastive_loss, reconstruction_loss = forward(eval_batch)
            eval_loss += loss.item()
            eval_contrastive_loss += contrastive_loss.item()
            eval_reconstruction_loss += reconstruction_loss.item()

    eval_loss /= len(eval_dataloader)
    eval_contrastive_loss /= len(eval_dataloader)
    eval_reconstruction_loss /= len(eval_dataloader)

    if verbose:
        print(f"Epoch: {epoch}, Step: {global_step}, Eval Loss: {eval_loss}")
    accelerator.log({
        "eval/loss": eval_loss,
        "eval/contrastive_loss": eval_contrastive_loss,
        "eval/reconstruction_loss": eval_reconstruction_loss,
    })

    ## Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the provided model on the compression loss.")
    parser.add_argument(
        '--model',
        help=("The model to pretrain. Default is 'cayjobla/bert-base-uncased-reduced'."),
        type=str,
        default=None,
        dest="model_name"
    )
    parser.add_argument(
        '--revision',
        help=("The revision of the model to use. Default is 'main'"),
        type=str,
        default="main"
    )
    parser.add_argument(
        '--dataset',
        '--dataset_path',
        help=("The dataset to train the model on. Required for training."),
        type=str,
        default=None,
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
        '--tokenizer',
        help=("The path to tokenizer to use during training. Default is to use the tokenizer of the model we are training."),
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
    # parser.add_argument(
    #     '--eval_strategy',
    #     help=("The strategy to use for evaluation during training. Default is 'steps'."),
    #     type=str,
    #     default="steps"
    # )
    parser.add_argument(
        '--eval_steps',
        help=("The number of steps between evaluations during training. Default is 50000."),
        type=int,
        default=50000
    )
    # parser.add_argument(
    #     '--save_strategy',
    #     help=("The strategy to use for saving during training. Default is 'steps'."),
    #     type=str,
    #     default="steps"
    # )
    parser.add_argument(
        '--save_steps',
        help=("The number of steps between saves during training. Default is 50000."),
        type=int,
        default=50000
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
        '--logging_steps',
        help=("The number of steps between logging during training. Default is 50."),
        type=int,
        default=50
    )
    # parser.add_argument(
    #     '--no_contrast',
    #     help=("Indicates that contrastive loss should not be used during training. Default is to use contrastive loss."),
    #     default=True,
    #     action="store_false",
    #     dest="do_contrast"
    # )
    # parser.add_argument(
    #     '--no_reconstruction',
    #     help=("Indicates that reconstruction loss should not be used during training. Default is to use reconstruction loss."),
    #     default=True,
    #     action="store_false",
    #     dest="do_reconstruction"
    # )
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the trained model to. If None, the model name is used."),
        type=str,
        default=None
    )   
    # parser.add_argument(
    #     '--push_to_hub',
    #     help=("Indicates that the trainer should push the trained model to the hub. Default is False."),
    #     default=False,
    #     action="store_true"
    # )
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
    # parser.add_argument(
    #     '--disable_tqdm',
    #     help=("Indicates that tqdm should be disabled. Default is False."),
    #     default=False,
    #     action="store_true"
    # )
    parser.add_argument(
        '--verbose',
        '-v',
        help=("Whether to print preprocessing progress information. Default is False."),
        default=False,
        action="store_true"
    )    
    # parser.add_argument(
    #     '--checkpoint',
    #     help=("The checkpoint to resume training from. Default is None."),
    #     type=str,
    #     default=None
    # )  

    kwargs = parser.parse_args()
    train(**vars(kwargs))