# evaluate_glue.py

from transformers import (
    set_seed, 
    AutoConfig, 
    AutoTokenizer,
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    default_data_collator, 
    TrainerCallback
)
from datasets import load_dataset
import evaluate
import numpy as np
import argparse
import os

import reduced_encoders # Load reduced models into the transformers auto classes


task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
task_to_name = {
    "cola": "CoLA",
    "mnli": "MNLI-m",
    'mnli-mm': "MNLI-mm",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp": "QQP",
    "rte": "RTE",
    "sst2": "SST-2",
    "stsb": "STS-B",
    "wnli": "WNLI",
    "ax": "AX"
}


def evaluation(
        model,
        task, 
        *,
        revision="main",
        batch_size=16, 
        learning_rate=3e-5, 
        num_epochs=3, 
        logging_steps=50, 
        output_dir=None,
        run_name=None, 
        seed=None, 
        verbose=False, 
        do_predict=False,
        disable_tqdm=False,
    ):
    ## Set seed
    if seed is not None:
        set_seed(seed)

    ## Load preprocessed data
    if verbose: 
        print(f"Loading the '{task}' split of the GLUE dataset...")
    task = task.lower()
    data = load_dataset("glue", task)

    if task == "mnli" and do_predict is True:
        data["ax"] = load_dataset("glue", "ax")["test"]

    # Get number of classification labels
    is_regression = task == "stsb"
    if not is_regression:
        label_list = data["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    ## Load config
    if model is None:
        raise ValueError("Please specify a model to evaluate.")
    model_name = model
    config = AutoConfig.from_pretrained(model_name, revision=revision, num_labels=num_labels)

    ## Load tokenizer
    if verbose: 
        print(f"Loading {model_name} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

    ## Load model
    if verbose: 
        print(f"Loading {model_name} model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, config=config, ignore_mismatched_sizes=True
    )

    ## Define training arguments
    if verbose: 
        print(f"Defining training arguments...")
    if output_dir is None: output_dir = model_name.split("/")[-1]
    if run_name is None: run_name = f"eval-glue-{task}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="no",
        push_to_hub=False,
        logging_steps=logging_steps,
        run_name=run_name,
        disable_tqdm=disable_tqdm,
    )

    ## Preprocess data
    if verbose: 
        print(f"Preprocessing the dataset for the '{task}' task...")
    sentence1_key, sentence2_key = task_to_keys[task]
    padding = "max_length"

    def preprocess_function(examples):
        # Tokenize the texts
        if sentence2_key is None:
            args = (examples[sentence1_key],)
        else:
            args = (examples[sentence1_key], examples[sentence2_key])
        result = tokenizer(*args, padding=padding, max_length=128, 
                            truncation=True)   
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        data = data.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on dataset",
        )

    # Define datasets
    train_dataset = data["train"]
    eval_dataset = data["validation_matched" if task=="mnli" else "validation"]

    ## Load evaluation metric
    if verbose: 
        print(f"Load the evaluation metric for the '{task}' task...")
    metric = evaluate.load("glue", task)

    def compute_metrics(p):
        preds = p.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Initialize our Trainer
    if verbose: 
        print(f"Initialize Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    ## Add evaluation for 0th epoch
    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            if state.global_step == 0:
                control.should_evaluate = True
    trainer.add_callback(EvaluateFirstStepCallback())

    ## Train the model
    if verbose: 
        print(f"Train the model...")
    train_result = trainer.train()

    ## Save the model
    if verbose: 
        print(f"Save the model...")
    metrics = train_result.metrics
    trainer.save_model(output_dir=output_dir)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    if verbose: 
        print(f"Evaluate the model on the validation set...")
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [task]
    eval_datasets = [eval_dataset]
    if task == "mnli":
        tasks.append("mnli-mm")
        eval_datasets.append(data["validation_mismatched"])
        combined = {}

    for eval_dataset, subtask in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        metrics["eval_samples"] = len(eval_dataset)

        if subtask == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if subtask is not None and "mnli" in subtask:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if "mnli" in subtask else metrics)

    if do_predict:
        if verbose:
            print("Predict the test set labels...")
        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [task]
        predict_dataset = data["test_matched" if task == "mnli" else "test"]
        predict_datasets = [predict_dataset]
        if task == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(data["test_mismatched"])
            tasks.append("ax")                          # Also add 'ax' task
            predict_datasets.append(data["ax"])

        for predict_dataset, subtask in zip(predict_datasets, tasks):
            predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(
                            predict_dataset, 
                            metric_key_prefix="predict"
                        ).predictions

            if is_regression:
                predictions = np.squeeze(predictions)
            else:
                predictions = np.argmax(predictions, axis=1)

            output_predict_file = os.path.join(output_dir, 
                                                f"{task_to_name[subtask]}.tsv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a trained reduced encoder on the GLUE dataset."
    )
    parser.add_argument(
        '--model',
        required=True,
        help=("The path to the model to evaluate."),
        type=str,
    )
    parser.add_argument(
        '--task',
        required=True,
        help=("The glue task to train/evaluate on. Required for training."),
        type=str,
    )
    parser.add_argument(
        '--revision',
        help=("The revision or branch of the model to use. Default is 'main'"),
        type=str,
        default="main"
    )
    parser.add_argument(
        '--batch_size',
        help=("The batch size to use during training. Default is 16."),
        type=int,
        default=16
    )
    parser.add_argument(
        '--learning_rate',
        help=("The learning rate to use during training. Default is 3e-5."),
        type=float,
        default=3e-5
    )
    parser.add_argument(
        '--num_epochs',
        help=("The number of epochs to train for. Default is 3."),
        type=int,
        default=3
    )
    parser.add_argument(
        '--logging_steps',
        help=("The number of steps between logging during training. "
                "Default is 50."),
        type=int,
        default=50
    )
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the trained model to. If None, "
                "the model name is used."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--run_name',
        help=("The name of the WandB run. Default is 'eval-glue-{task}'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--seed',
        help=("The seed to use for randomness in preprocessing."),
        type=int,
        default=None
    )
    parser.add_argument(
        '--verbose',
        '-v',
        help=("Whether to print preprocessing progress information. "
                "Default is False."),
        default=False,
        action="store_true"
    )  
    parser.add_argument(
        '--do_predict',
        help=("Whether to predict on the test set after fine-tuning. "
                "Default is False."),
        default=False,
        action="store_true"
    )
    parser.add_argument(
        '--disable_tqdm',
        help=("Whether to disable tqdm progress bars. Default is False."),
        default=False,
        action="store_true"
    )

    kwargs = parser.parse_args()
    evaluation(**vars(kwargs))