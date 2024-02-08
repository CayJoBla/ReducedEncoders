# train_hyperopt_compressed.py

import argparse

from datasets import load_dataset
from transformers import set_seed, AutoTokenizer, DataCollatorWithPadding, TrainingArguments
from custom_trainer import CustomTrainer
from reduced_encoders import MPNetCompressedForPretraining

def hyperopt(model=None, dataset_path=None, dataset_name=None, split="train", tokenizer=None, revision="main", train_size=None, 
             validation_size=None, logging_steps=50, num_epochs=1, output_dir=None, seed=None, verbose=False):
    ## Set seed
    if seed is not None:
        set_seed(seed)

    ## Set model name
    model_name = model or "cayjobla/all-mpnet-base-v2-compressed"

    ## Load tokenizer
    if tokenizer is None: tokenizer = model_name
    if verbose: print(f"Loading {tokenizer} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision)

    ## Load data for preprocessing
    if dataset_path is None: 
        raise ValueError("The 'dataset_path' argument must be provided.")
    full_dataset_path = f"{dataset_path}:{dataset_name}" if dataset_name else dataset_path

    if verbose: print(f"Loading the '{split}' split of the '{full_dataset_path}' dataset...")
    dataset = load_dataset(dataset_path, dataset_name, split=split)

    ## Preprocess the data
    def preprocess_data(batch):
        return tokenizer(batch["text"], truncation=True)    # No padding, we pad in the data collator

    if verbose: print(f"Preprocessing the '{split}' split of the '{full_dataset_path}' dataset...")
    preprocessed = dataset.map(preprocess_data, batched=True, batch_size=1000, remove_columns=dataset.column_names)

    ## Create a train and validation split of the preprocessed data    
    if verbose: print(f"Creating a train and validation split of the preprocessed data...")
    if train_size is None and validation_size is None: # train_test_split should handle other cases or 
        train_size = 0.9 
    input_data = preprocessed.train_test_split(train_size=train_size, test_size=validation_size, seed=seed)

    ## Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ## Define training arguments
    if output_dir is None: output_dir = model_name.split("/")[-1]
    run_name = run_name or f"hyperopt-compressed-{dataset_path}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="no",
        num_train_epochs=num_epochs,
        push_to_hub=False,
        logging_steps=logging_steps,
    )

    ## Set up hyperparameter optimization requirements
    def optuna_hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32, 64]),
            "per_device_eval_batch_size": trial.suggest_categorical("per_device_eval_batch_size", [8, 16, 32, 64]),
            "alpha": trial.suggest_float("alpha", 0, 1),
            "beta": trial.suggest_float("beta", 0, 1)
        }
        
    
    def model_init(trial):
        # Init model
        model = MPNetCompressedForPretraining.from_pretrained(
            model_name,
            revision=revision,
            alpha= 1 if trial is None else trial.params["alpha"],
            beta= 1 if trial is None else trial.params["beta"],
            )

        ## Freeze base model parameters
        for param in model.base_model.parameters():
            param.requires_grad = False

        return model

    def compute_objective(metrics):
        return metrics["eval_contrastive_loss"] + metrics["eval_reconstruction_loss"]

    ## Define our custom trainer
    trainer = CustomTrainer(
        model=None,
        model_init=model_init,
        args=training_args,
        data_collator=data_collator,
        train_dataset=input_data["train"],
        eval_dataset=input_data["test"],
        tokenizer=tokenizer,
        extra_loss_index_mapping={"contrastive_loss": 0, "reconstruction_loss": 1},
        do_initial_eval=False,
    )

    best_trial = trainer.hyperparameter_search(
        direction="minimize",
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=3,
        compute_objective=compute_objective,
    )

    print("Best trial:", best_trial)

    return best_trial



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
    hyperopt(**vars(kwargs))