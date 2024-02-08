# train_compressed.py

from transformers import set_seed, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
from reduced_encoders import MPNetCompressedForPretraining
from datasets import load_dataset
import argparse
from functools import partial

from custom_trainer import CustomTrainer

def train(model=None, dataset_path=None, dataset_name=None, split="train", tokenizer=None, revision="main", train_size=None, 
          validation_size=None, batch_size=16, eval_strategy="steps", eval_steps=50000, save_strategy="steps", save_steps=50000, 
          logging_steps=50, learning_rate=2e-4, num_epochs=1, alpha=1., beta=1., checkpoint=None, output_dir=None, push_to_hub=False, 
          run_name=None, seed=None, verbose=False):
    ## Set seed
    if seed is not None:
        set_seed(seed)

    ## Load model
    model_name = model or "cayjobla/all-mpnet-base-v2-compressed"
    if verbose: print(f"Loading {model_name} model...")
    model = MPNetCompressedForPretraining.from_pretrained(
                model_name, 
                revision=revision,
                alpha=alpha, 
                beta=beta
                )

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

    ## Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    ## Define training arguments
    if output_dir is None: output_dir = model_name.split("/")[-1]
    if run_name is None: run_name = "pretrain-compressed-" + dataset_path
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
        run_name=run_name,
    )

    # Initialize custom Trainer and metrics to log contrastive and reconstruction losses
    # def compute_metrics(eval_pred, loss_index_mapping=None):
    #     model_output = eval_pred.predictions
    #     out_dict = {}
    #     if loss_index_mapping is not None:
    #         for k, v in loss_index_mapping.items():
    #             out_dict[k] = model_output[v].mean().item()
    #     return out_dict

    extra_loss_index_mapping = {"contrastive_loss": 0, "reconstruction_loss": 1}    # TODO: This should be pulled from the model

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=input_data["train"],
        eval_dataset=input_data["test"],
        tokenizer=tokenizer,
        # extra_losses=list(extra_loss_index_mapping.keys()),
        # compute_metrics=partial(compute_metrics, loss_index_mapping=extra_loss_index_mapping),
        extra_loss_index_mapping=extra_loss_index_mapping,
        do_initial_eval=True,
    )

    # ## Add initial evaluation
    # class EvaluateFirstStepCallback(TrainerCallback):
    #     def on_step_begin(self, args, state, control, **kwargs):
    #         if state.global_step == 0:
    #             control.should_evaluate = True

    # trainer.add_callback(EvaluateFirstStepCallback())

    ## Train the model
    if checkpoint == "latest": checkpoint = True
    trainer.train(resume_from_checkpoint=checkpoint)

    ## Do final evaluation
    final_metrics = trainer.evaluate()

    ## Save the model
    trainer.save_model(output_dir=output_dir)

    return final_metrics


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
        '--alpha',
        help=("The weight given to the constrastive loss during training. Default is 1."),
        type=float,
        default=1
    )
    parser.add_argument(
        '--beta',
        help=("The weight given to the reconstruction loss during training. Default is 1."),
        type=float,
        default=1
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
    parser.add_argument(
        '--checkpoint',
        help=("The checkpoint to resume training from. Default is None."),
        type=str,
        default=None
    )

    kwargs = parser.parse_args()
    train(**vars(kwargs))