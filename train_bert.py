# bert_reduced_train.py

# Torch imports
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

# HF imports
from transformers import BertTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, default_data_collator, get_scheduler
from datasets import load_dataset
from accelerate import Accelerator
from huggingface_hub import get_full_repo_name, Repository

from bert_reduced import BertReducedForMaskedLM
from tqdm.auto import tqdm
import math
import argparse

def train(model=None, dataset=None, num_shards=None, index=0, train_size=None, test_size=None, batch_size=64, num_epochs=1, 
          learning_rate=5e-5, tol=1e-8, finetune_base=False, scheduler_type="linear", num_warmup_steps=0, logging_steps=1000, 
          logging_strat="epoch", output_dir=None, push_to_hub=True, repo_name=None, wandb_project=None, run_name=None):

    def get_dataloaders(dataset, num_shards, index, train_size, test_size, batch_size):
        print("Load data...")
        lm_dataset = load_dataset(dataset)  

        # Shard the dataset
        if not (num_shards is None or num_shards == 1 or num_shards == 0):
            print(f"Selecting data shard {index+1}/{num_shards}...")
            lm_dataset["train"] = lm_dataset["train"].shard(num_shards, index)
            lm_dataset["test"] = lm_dataset["test"].shard(num_shards, index)
        else:
            num_shards = 1
            index = 0
        
        if not (train_size is None and test_size is None):
            print("Downsample the data...")
            seed = 42
        if train_size is None:
            train_data = lm_dataset["train"]
        else:
            train_data = lm_dataset["train"].train_test_split(train_size=train_size, seed=seed)["train"]
        if train_size is None:
            test_data = lm_dataset["test"]
        else:
            test_data = lm_dataset["test"].train_test_split(test_size=test_size, seed=seed)["test"]

        lm_dataset["train"] = train_data
        lm_dataset["test"] = test_data

        # Create the training data collator
        mlm_data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15, return_tensors="pt")

        # Create the dataloaders
        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size, collate_fn=mlm_data_collator)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, collate_fn=default_data_collator)

        return train_dataloader, test_dataloader, num_shards, index

    def evaluate(dataloader, progress_bar=None):
        """Evaluate the validation set for logging"""
        model.eval()
        losses = []
        for batch in dataloader:
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))
            if progress_bar: progress_bar.update(1)
        losses = torch.cat(losses)
        losses = losses[:len(dataloader)]
        mean_loss = torch.mean(losses)
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError:
            perplexity = float("inf")
        model.train()
            
        return mean_loss, perplexity
    
    def save_model(model, output_dir, repo=None):
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
            if push_to_hub:
                commit_message = ""
                if logging_strat=="epoch":
                    commit_message = f"Epoch: {epoch+1}/{num_epochs}, Shard: {index+1}/{num_shards}"
                elif logging_strat=="step":
                    commit_message = f"Global Step: {global_step}/{num_epochs*num_update_steps_per_epoch}, Shard: {index+1}/{num_shards}"
                repo.push_to_hub(
                    commit_message=commit_message, 
                    blocking=False
                )

    # Set default model and dataset
    if model is None: model = "cayjobla/bert-base-uncased-reduced"
    if dataset is None: dataset = "cayjobla/wikipedia_tokenized"

    model_checkpoint = model
    model_name = model_checkpoint.split("/")[-1]

    ## Load the latest reduced model
    print("Load model...")
    try:
        model = BertReducedForMaskedLM.from_pretrained(model_checkpoint)
    except:
        print("Could not load BertReducedForMaskedLM model, attempting load with AutoModelForMaskedLM...")
        model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    
    ## Get dataloaders
    train_dataloader, test_dataloader, num_shards, index = get_dataloaders(
        dataset, num_shards, index, train_size, test_size, batch_size
    )

    ## Set up for training
    print("Prep for training...")
    # Freeze the Bert model so its parameters aren't updated
    if not finetune_base:
        print("Freeze the base model weights...")
        for param in model.base_model.parameters():
            param.requires_grad = False
    
    # Define the optimizer for training
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=tol)

    # Define the Accelerator
    do_log = wandb_project is not None
    if do_log:
        print(f"Logging to WandB project {wandb_project}...\n")
        accelerator = Accelerator(log_with="wandb")
        init_kwargs= {"wandb":{"name":run_name}} if run_name else {}
        accelerator.init_trackers(
            project_name=wandb_project, 
            init_kwargs=init_kwargs
        )
        print("")
    else:
        accelerator = Accelerator()

    # Prep objects with accelerator
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    # Define training arguments
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_epochs * num_update_steps_per_epoch

    # Get learning rate scheduler
    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Set up HF Hub paramters
    output_dir = model_name if output_dir is None else output_dir
    if push_to_hub:
        print("Set up HuggingFace hub parameters...")
        if repo_name is None: repo_name = get_full_repo_name(model_name)
        repo = Repository(output_dir, clone_from=repo_name)
    else:
        repo = None

    # Print training setup to console
    print("Model name:", model_name)
    print("Training data batches:", len(train_dataloader))
    print("Evaluation data batches:", len(test_dataloader))
    print(f"Data shard: {index+1}/{num_shards}")
    print("Batch size:", batch_size)
    print("Num epochs:", num_epochs)
    print("Update steps per epoch:", num_update_steps_per_epoch)
    print("Total training steps:", num_training_steps)
    print("Fine-tuning base model:", finetune_base)
    print("Learning rate:", learning_rate)
    print("Tolerance (epsilon):", tol)
    print("Learning rate scheduler:", scheduler_type)
    print("Warmup steps:", num_warmup_steps)
    print("Logging by:", logging_strat)
    if logging_strat == "step": print("Logging steps:", logging_steps)
    print("Model output directory:", output_dir)
    print("Push to HuggingFace hub:", push_to_hub)
    if push_to_hub: print("Repository name:", repo_name)
    print("Logging with WandB:", wandb_project is not None)
    if wandb_project is not None:
        print("Project name:", wandb_project)
        print("Run name:", accelerator.trackers[0].run._settings.run_name)
    print("")

    # Record initial evaluation loss
    progress_bar = tqdm(range(len(test_dataloader)), desc=">>> Initial Evaluation")
    mean_loss, perplexity = evaluate(test_dataloader, progress_bar=progress_bar)
    if do_log: accelerator.log({"eval_loss":mean_loss, "eval_perplexity":perplexity}, step=0)
    progress_bar.close()

    ## Training loop
    print("Start training...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model.train()
        progress_bar = tqdm(range(num_update_steps_per_epoch), desc=">>> Training")
        for i, batch in enumerate(train_dataloader):
            # Training
            global_step = epoch*num_update_steps_per_epoch + i + 1

            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if do_log: accelerator.log({"train_loss":loss}, step=global_step)
            progress_bar.update(1)

            # Evaluation
            if (logging_strat=="step" and global_step%logging_steps==0) or (logging_strat=="epoch" and i+1==num_update_steps_per_epoch):
                eval_progress_bar = tqdm(range(len(test_dataloader)), desc=">>> Evaluation")
                mean_loss, perplexity = evaluate(test_dataloader, progress_bar=eval_progress_bar)
                if do_log: accelerator.log({"eval_loss":mean_loss, "eval_perplexity":perplexity}, step=global_step)
                eval_progress_bar.close()
                save_model(model, output_dir, repo)     # Save model

        progress_bar.close()    # Finish the epoch

    # Final Evaluation
    if logging_strat=="step":   # If logging_strat = "epoch", we would have evaluated and saved already
        eval_progress_bar = tqdm(range(len(test_dataloader)), desc=">>> Final Evaluation")
        mean_loss, perplexity = evaluate(test_dataloader, progress_bar=eval_progress_bar)
        if do_log: accelerator.log({"eval_loss":mean_loss, "eval_perplexity":perplexity}, step=num_training_steps)
        eval_progress_bar.close()
        save_model(model, output_dir, repo)   # Save model

    if do_log: accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a reduced BERT model for masked language modeling")
    parser.add_argument(
        '--model',
        '-m',
        help=("The checkpoint of the model to use for training. Default is cayjobla/bert-base-uncased-reduced."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--dataset',
        '-d',
        help=("The dataset to use for training on the model. Default is the wikipedia dataset."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--num_shards',
        '-k',
        help=("The number of shards to split the data into. Training is only done on one of the shards according to the given index. Default is 1 shard, or no sharding"),
        type=int,
        default=1
    )  
    parser.add_argument(
        '--index',
        '-i',
        help=("The index of the shard of data to train on if sharding is used. Default is 0."),
        type=int,
        default=0
    )  
    parser.add_argument(
        '--train_size',
        '-a',
        help=("The proportion of the training data shard to use during training. Default is the entire training data shard selected."),
        type=float,
        default=None
    )   
    parser.add_argument(
        '--test_size',
        '-c',
        help=("The proportion of the test data shard to use for evaluation. Default is the entire test data shard selected."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--batch_size',
        '-b',
        help=("The batch size to use during training. Default is 64."),
        type=int,
        default=64
    )
    parser.add_argument(
        '--finetune_base',
        '-f',
        help=("Indicates that the base BERT model should be finetuned alongside the model head and reduction layers. Default is to freeze the base model weights."),
        default=False,
        action="store_true"
    )  
    parser.add_argument(
        '--learning_rate', 
        '-lr',
        help=("The learning rate to use in the optimizer. Default is 5e-5."),
        type=float,
        default=5e-5
    )
    parser.add_argument(
        '--num_epochs',
        '-n',
        help=("The number of epochs to use in training. Default is 1."),
        type=int,
        default=1
    )
    parser.add_argument(
        '--scheduler_type',
        '-s',
        help=("The name of the type of learning rate scheduler to use in optimization. Default is 'linear'."),
        type=str,
        default="linear"
    )
    parser.add_argument(
        '--num_warmup_steps',
        '-w',
        help=("The number of warmup steps for the learning rate scheduler. Default is 0."),
        type=int,
        default=0
    )
    parser.add_argument(
        '--logging_steps',
        '-ls',
        help=("The number of global optimization steps between logging. Default is 1000."),
        type=int,
        default=1000
    )
    parser.add_argument(
        '--logging_strat',
        '-l',
        help=("The strategy to use for logging, either 'epoch' or 'step'. Default is 'epoch'"),
        type=str,
        default="epoch"
    )
    parser.add_argument(
        '--tol',
        '-t',
        help=("The tolerance for convergence in the optimizer. Default is 1e-8."),
        type=float,
        default=1e-8
    )
    parser.add_argument(
        '--no_push',
        '-np',
        help=("Indicates that the training model should not be pushed to the HuggingFace hub. Default is False."),
        dest='push_to_hub',
        default=True,
        action="store_false"
    )   
    parser.add_argument(
        '--repo_name',
        '-repo',
        help=("The name of the HuggingFace repository to push to. Defaults to the name of the model."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--output_dir',
        '-o',
        help=("The output directory to save the model to. Defaults to the name of the model."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--wandb_project',
        '-proj',
        help=("The name of the Weights and Biases project to push logging to. If none is specified, WandB is not used for logging."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--run_name',
        '-run',
        help=("The name to give the training run if WandB logging is used. If none is specified, let WandB assign a random run name"),
        type=str,
        default=None
    )

    kwargs = parser.parse_args()

    train(**vars(kwargs))
