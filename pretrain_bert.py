# pretrain_bert.py

from transformers import set_seed, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from bert_reduced import BertReducedForPreTraining
from datasets import load_dataset
import argparse

def pretrain(model=None, dataset=None, split="train", tokenizer=None, revision="main", train_size=None, 
             validation_size=None, mlm_probability=0.15, batch_size=16, output_dir=None, eval_strategy="steps", 
             eval_steps=50000, save_strategy="steps", save_steps=50000, learning_rate=2e-4, num_epochs=1, 
             logging_steps=50, push_to_hub=False, run_name=None, seed=42, verbose=False):
    ## Set seed
    set_seed(seed)

    ## Load model
    model_name = "cayjobla/bert-base-uncased-reduced" if model is None else model
    if verbose: print(f"Loading {model_name} model...")
    model = BertReducedForPreTraining.from_pretrained(model_name, revision=revision)

    ## Load tokenizer
    if tokenizer is None: tokenizer = model_name
    if verbose: print(f"Loading {tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision)

    ## Load preprocessed data
    if dataset is None: raise ValueError("The 'dataset' argument must be provided.")
    if verbose: print(f"Loading the '{split}' split of the '{dataset}' dataset...")
    preprocessed = load_dataset(dataset, split)["train"]

    ## Add labels column to the data if it doesn't exist
    if "labels" not in preprocessed.column_names:
        preprocessed["labels"] = preprocessed.input_ids.detach().clone()

    ## Create a train and validation split of the preprocessed data
    if train_size is None and validation_size is None: train_size = 0.9
    if verbose: print(f"Creating a train and validation split of the preprocessed data...")
    input_data = preprocessed.train_test_split(train_size=train_size, test_size=None, seed=seed)

    ## Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability)

    ## Define training arguments
    if output_dir is None: output_dir = model_name.split("/")[-1]
    if run_name is None: run_name = "mlm-nsp-" + dataset
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=push_to_hub,
        logging_steps=logging_steps,
        run_name=run_name
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=input_data["train"],
        eval_dataset=input_data["test"],
        tokenizer=tokenizer,
    )

    ## Train the model
    trainer.train()

    ## Save the model
    trainer.save_model(output_dir=output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the provided model on the given preprocessed dataset for MLM and NSP.")
    parser.add_argument(
        '--model',
        help=("The model to pretrain. Default is 'cayjobla/bert-base-uncased-reduced'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--revision',
        help=("The revision of the model to use. Default is 'main'"),
        type=str,
        default="main"
    )
    parser.add_argument(
        '--dataset',
        help=("The dataset to train the model on. Required for training."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--split',
        help=("The split of the dataset to use. Default is the '20220301.en' split of the wikipedia dataset."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--tokenizer',
        help=("The tokenizer to use during training. Default is to use the model's tokenizer."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--train_size',
        help=("The size of the data to use in training. Default is None."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--validation_size',
        help=("The size of the validation set when evaluating during training. Default is None."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--mlm_probability',
        help=("The probability of masking a token for MLM. Default is 0.15."),
        type=float,
        default=0.15
    )
    parser.add_argument(
        '--batch_size',
        help=("The batch size to use during training. Default is 16."),
        type=int,
        default=16
    )
    parser.add_argument(
        '--eval_strategy',
        help=("The strategy to use for evaluation during training. Default is 'steps'."),
        type=str,
        default="steps"
    )
    parser.add_argument(
        '--eval_steps',
        help=("The number of steps between evaluations during training. Default is 50000."),
        type=int,
        default=50000
    )
    parser.add_argument(
        '--save_strategy',
        help=("The strategy to use for saving during training. Default is 'steps'."),
        type=str,
        default="steps"
    )
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
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the trained model to. If None, the model name is used."),
        type=str,
        default=None
    )   
    parser.add_argument(
        '--push_to_hub',
        help=("Indicates that the trainer should push the trained model to the hub. Default is False."),
        default=False,
        action="store_true"
    )
    parser.add_argument(
        '--run_name',
        help=("The name of the WandB run. Default is 'mlm-nsp-{dataset}'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--seed',
        help=("The seed to use for randomness in preprocessing. Default is 42."),
        type=int,
        default=42
    )
    parser.add_argument(
        '--verbose',
        '-v',
        help=("Whether to print preprocessing progress information. Default is False."),
        default=False,
        action="store_true"
    )    

    kwargs = parser.parse_args()
    pretrain(**vars(kwargs))