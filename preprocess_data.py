# preprocess_data.py

from transformers import set_seed, AutoTokenizer
from datasets import load_dataset, concatenate_datasets
import random
import math
import argparse
import torch

def preprocess(dataset=None, split=None, tokenizer=None, revision="main", shard_size=100_000,
               train_size=None, test_size=0.1, seed=42, output_dir=None, verbose=False):
    ## Set seed
    set_seed(seed)

    ## Load tokenizer
    if verbose: print("Load tokenizer...")
    if tokenizer is None: tokenizer = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, revision=revision)

    ## Load data and preprocess
    if dataset is None: dataset = "wikipedia"
    if split is None and dataset == "wikipedia": split = "20220301.en"

    if verbose: print(f"Loading the '{split}' split of the '{dataset}' dataset...")
    data = load_dataset(dataset) if split is None else load_dataset(dataset, split)

    ## Create bag of sentences
    def get_bag(batch):
       bag = [sentence.strip() 
              for article in batch["text"] 
              for paragraph in article.split("\n") 
              for sentence in paragraph.split(".")    
              if sentence.strip() != "" and paragraph.count(".") > 0]
       return {"bag":bag}

    bag = data.map(
        get_bag, 
        batched=True,
        remove_columns=['id','url','title','text'],
        desc="Creating bag of sentences",
    )["train"]["bag"]
    bag_size = len(bag)

    def preprocess(batch):
        random.seed(seed)
        sentence1 = []
        sentence2 = []
        labels = []

        for article in batch["text"]:
            for paragraph in article.split("\n"):
                sentences = [sentence.strip() for sentence in paragraph.split(".") if sentence.strip() != ""]
                num_sentences = len(sentences)
                if num_sentences > 1:
                    start = random.randint(0, num_sentences-2)
                    sentence1.append(sentences[start])
                    if random.random() < 0.5:
                        sentence2.append(sentences[start+1])
                        labels.append(0)
                    else:
                        sentence2.append(bag[random.randint(0, bag_size-1)])
                        labels.append(1)
        
        inputs = tokenizer(sentence1, sentence2, return_tensors="pt", max_length=512, 
                           truncation=True, padding="max_length")
        inputs["next_sentence_label"] = torch.LongTensor(labels)
        inputs["labels"] = inputs.input_ids.detach().clone()
        return inputs

    def sharded_preprocess(dataset, shard_size=100_000):
        num_shards = math.ceil(len(dataset["train"]) / shard_size)
        shards = []
        for i in range(num_shards):
            if i == num_shards-1:
                data_shard = dataset["train"]
            else:
                dataset = dataset["train"].train_test_split(test_size=min(shard_size, len(dataset["train"])), shuffle=False)
                data_shard = dataset["test"]
            shards.append(data_shard.map(
                preprocess, 
                batched=True, 
                remove_columns=['id', 'url', 'title', 'text'], 
                desc=f"Preprocess shard {i+1} of the dataset for NSP"
            ))
        return concatenate_datasets(shards, split="train")

    ## Preprocess the data
    if verbose: print("Preprocess the data...")
    processed_data = sharded_preprocess(data, shard_size=shard_size)

    ## Split into train/test sets
    if verbose: print("Split data into train and test sets")
    input_data = processed_data.train_test_split(train_size=train_size, test_size=test_size, seed=seed)

    ## Save dataset to disk
    if output_dir is None:
        output_dir = f"{dataset}-pretrain-processed"
    input_data.save_to_disk(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize, chunk, and split the dataset. Randomly mask the test data. Push the dataset to the HuggingFace hub.")
    parser.add_argument(
        '--dataset',
        help=("The dataset to preprocess. Default is the wikipedia dataset."),
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
        help=("The tokenizer to use for preprocessing. Default is 'bert-base-uncased'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--revision',
        help=("The revision of the tokenizer to use. Default is 'main'"),
        type=str,
        default="main"
    )
    parser.add_argument(
        '--shard_size',
        help=("The size of shard to use in data preprocessing. Larger sizes run faster, smaller sizes cache more often. Default is 100_000."),
        type=int,
        default=100_000
    )
    parser.add_argument(
        '--train_size',
        help=("The size of the training set when splitting the data. Default is None."),
        type=float,
        default=None
    )
    parser.add_argument(
        '--test_size',
        help=("The size of the test set when splitting the data. Default is 0.1."),
        type=float,
        default=0.15
    )
    parser.add_argument(
        '--seed',
        help=("The seed to use for randomness in preprocessing. Default is 42."),
        type=int,
        default=None
    )
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the preprocessed data to. If None, a default directory is chosen."),
        type=str,
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

    preprocess(**vars(kwargs))