# preprocess_data.py

import argparse
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from huggingface_hub import get_full_repo_name


def preprocess(dataset=None, split=None, tokenizer=None, train_size=None, test_size=0.15, chunk_size=64, 
               mlm_probability=0.15, repo_name=None):
    ## Load tokenizer
    print("Load tokenizer...")
    if tokenizer is None: tokenizer = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    ## Load data and preprocess
    # Define default values
    if dataset is None: dataset = "wikipedia"
    if split is None and dataset == "wikipedia": split = "20220301.en"
    dataset_name = dataset.split("/")[-1]

    print(f"Loading the {split} split of the {dataset} dataset...")
    data = load_dataset(dataset) if split is None else load_dataset(dataset, split) # Load
    data = data.shuffle(seed=42)            # Shuffle the dataset

    ## Tokenize the data
    def tokenize(batch):
        return tokenizer(batch["text"])

    print("Tokenize the dataset...")
    tokenized_dataset = data.map(tokenize, batched=True, remove_columns=["id", "url", "title", "text"])

    ## Chunk the data
    def chunk_tokens(batch):
        """Concatenate the samples in the batch and split into chunks of equal length"""
        concatenated_batch = {k: sum(batch[k], []) for k in batch.keys()}  # Concatenate
        total_length = len(concatenated_batch[list(batch.keys())[0]])  # Get total length
        total_length = (total_length // chunk_size) * chunk_size  # Drop the last chunk if smaller
        result = {
            k: [t[i:i+chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_batch.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    print("Chunk the data...")
    lm_dataset = tokenized_dataset.map(chunk_tokens, batched=True)

    # Shuffle and split into train-test sets
    print("Split the data into train and test sets...")
    lm_dataset = lm_dataset["train"].train_test_split(train_size=train_size, 
                                                      test_size=test_size, 
                                                      seed=42)  

    # Mask the test data once to reduce randomness at each evaluation
    def insert_random_mask(batch):
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=mlm_probability, return_tensors="pt")
        features = [dict(zip(batch, t)) for t in zip(*batch.values())]
        masked_inputs = data_collator(features)
        return {"masked_" + k: v for k, v in masked_inputs.items()}

    print("Add random mask to test set...")
    column_names = lm_dataset['test'].column_names
    eval_dataset = lm_dataset['test'].map(insert_random_mask, batched=True, remove_columns=column_names)
    eval_dataset = eval_dataset.rename_columns({"masked_" + column: column for column in column_names})

    # Cast the features to match the train dataset
    new_features = lm_dataset['train'].features.copy()
    eval_dataset = eval_dataset.cast(new_features)

    lm_dataset['test'] = eval_dataset
    
    print("Push dataset to hub...")
    if repo_name is None:
        repo_name = f"{dataset_name}-tokenized"
    repo_name = get_full_repo_name(repo_name)
    lm_dataset.push_to_hub(repo_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize, chunk, and split the dataset. Randomly mask the test data. Push the dataset to the HuggingFace hub.")
    parser.add_argument(
        '--dataset',
        '-d',
        help=("The dataset to preprocess. Default is the wikipedia dataset."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--split',
        '-s',
        help=("The split of the dataset to use. Default is the '20220301.en' split of the wikipedia dataset."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--tokenizer',
        '-t',
        help=("The tokenizer to use for preprocessing. Default is 'bert-base-uncased'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--train_size',
        '-a',
        help=("The size of the training set when splitting the data. Default is None."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--test_size',
        '-b',
        help=("The size of the test set when splitting the data. Default is 0.15."),
        type=float,
        default=0.15
    )
    parser.add_argument(
        '--chunk_size',
        '-c',
        help=("The number of tokens to include in each data chunk. Default is 64"),
        type=int,
        default=64
    )
    parser.add_argument(
        '--mlm_probability',
        '-p',
        help=("The probability to use when applying masking to the test set. Default is 0.15."),
        type=float,
        default=0.15
    )
    parser.add_argument(
        '--repo_name',
        '-r',
        help=("The HuggingFace repository to push the dataset to. Default is '{dataset_name}-tokenized'."),
        type=str,
        default=None
    )

    kwargs = parser.parse_args()

    preprocess(**vars(kwargs))