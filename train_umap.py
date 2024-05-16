# train_umap.py

from transformers import set_seed
from reduced_encoders import MPNetReducedConfig, MPNetCompressedModel
from datasets import load_dataset
from datasets.utils.logging import disable_progress_bar
import argparse
from umap_pytorch import PUMAP
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from tqdm import tqdm
import logging
import os


def train(model="cayjobla/all-mpnet-base-v2-compressed", dataset_path="wikipedia", dataset_name=None, 
            split="train", config=None, revision="main", train_size=None, validation_size=None, batch_size=16, 
            learning_rate=2e-4, num_epochs=1, n_neighbors=10, min_dist=0.1, metric="euclidean", checkpoint=None, 
            output_dir=None, do_encoding=None, seed=None, disable_tqdm=False, verbose=False):

    ## Set up logging
    logger = logging.getLogger("train_umap")
    logger.setLevel(logging.DEBUG)

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
        
    ## Set seed
    if seed is not None:
        set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = model

    ## Load the configuration
    if config is None: config = model_name
    if verbose: logger.debug(f"Loading the {config} model configuration...")
    config = MPNetReducedConfig.from_pretrained(config, revision=revision)

    ## Load model
    if verbose: logger.debug(f"Loading {model_name} model...")
    model = MPNetCompressedModel.from_pretrained(model_name, config=config, revision=revision)
    model.to(device)

    ## Freeze base model parameters
    for param in model.base_model.parameters():
        param.requires_grad = False

    ## Load data for preprocessing
    full_dataset_path = f"{dataset_path}:{dataset_name}" if dataset_name else dataset_path

    if verbose: logger.debug(f"Loading the '{split}' split of the '{full_dataset_path}' dataset...")
    dataset = load_dataset(dataset_path, dataset_name, split=split)

    ## Check whether the data needs to be encoded
    do_encoding = do_encoding or ("text" in dataset.column_names)
    if do_encoding and not ("text" in dataset.column_names):
        raise ValueError("do_encoding was specified as True, but the dataset does not include a 'text' column")
    elif not do_encoding and not ("data" in dataset.column_names):
        raise ValueError("The dataset does not have a 'data' column containing text embeddings. Provide a different dataset, or specify do_encoding=True")

    ## Preprocess the data (if necessary)
    if disable_tqdm:
        disable_progress_bar()

    if do_encoding:
        # Tokenize the dataset
        def tokenize_data(batch):
            return tokenizer(batch["text"], truncation=True, padding="max_length")

        if verbose: logger.debug(f"Tokenizing the '{split}' split of the '{full_dataset_path}' dataset...")
        tokenized = dataset.map(tokenize_data, batched=True, batch_size=1000, remove_columns=dataset.column_names)

        # Embed the dataset
        def embed_data(batch):
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            with torch.no_grad():
                outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = model.pooler(outputs[0], attention_mask) 
            return {"data": pooled_output.cpu().detach()}

        if verbose: logger.debug(f"Embedding the '{split}' split of the '{full_dataset_path}' dataset...")
        dataset = tokenized.map(embed_data, batched=True, batch_size=100, remove_columns=tokenized.column_names)
    
    embedding_dataset = dataset.with_format("torch")    # Convert to Pytorch tensor

    ## Create a train and validation split of the preprocessed data    
    if verbose: logger.debug(f"Creating a train and validation split of the embedding data...")
    if train_size is None and validation_size is None: # train_test_split function should handle all other cases 
        train_size = 0.9 
    input_data = embedding_dataset.train_test_split(train_size=train_size, test_size=validation_size, seed=seed)

    ## Downsample the data TODO: REMOVE THIS
    input_data["train"] = input_data["train"].select(range(3000000))

    ## Begin training the model
    if verbose: logger.debug("Training the reduction using ParametricUMAP...")

    # Define the PUMAP model
    pumap = PUMAP(
        encoder=model.reduce,
        decoder=model.expand,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=model.config.reduced_size,                 # The final reduced size
        beta=1.0,                                               # Default value, may need to tune
        reconstruction_loss=F.binary_cross_entropy_with_logits, # Default value
        lr=learning_rate,
        epochs=num_epochs,
        batch_size=batch_size,
        num_workers=127,
        num_gpus=1,
        match_nonparametric_umap=False,
    )

    # Fit the model
    os.environ['TOKENIZERS_PARALLELISM'] = "false"      # Disable warning for forking tokenizers parallelism
    torch.set_float32_matmul_precision('medium')
    pumap.fit(input_data["train"]["data"])
    os.unsetenv('TOKENIZERS_PARALLELISM')               # Unset the environment variable

    ## Save the model
    if output_dir is None:
        output_dir = model_name

    pumap_filename = f"{output_dir}/umap.pkl"
    if verbose: logger.debug(f"Saving the ParametricUMAP model to '{pumap_filename}'...")
    pumap.save(pumap_filename)

    if verbose: logger.debug(f"Saving the MPNetCompressedModel to '{output_dir}'...")
    model.to("cpu")
    model.reduce = pumap.model.encoder
    model.expand = pumap.model.decoder
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the dimensionality reduction of the provided model.")
    parser.add_argument(
        '--model',
        help=("The model to pretrain. Default is 'cayjobla/all-mpnet-base-v2-compressed'."),
        type=str,
        default="cayjobla/all-mpnet-base-v2-compressed"
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
        help=("The dataset to train the model on. Default is 'wikipedia' for the wikipedia dataset."),
        type=str,
        default="wikipedia",
        required=True,
        dest="dataset_path"
    )
    parser.add_argument(
        '--dataset_name',
        help=("The name of the data subset to use within the dataset_path dataset. Not required for all datasets. Default is None."),
        type=str,
        default=None,
    )
    parser.add_argument(
        '--split',
        help=("The split of the dataset to use for training. Default split is 'train'."),
        type=str,
        default="train"
    )
    parser.add_argument(
        '--config',
        help=("The path to model configuration to use during training. Default is to use the config of the provided model path."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--train_size',
        help=("The size of the training set to use. Default is 1-{validation_size} or 0.9."),
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
        '--n_neighbors',
        help=("The number of neighbors to use in the ParametricUMAP trainer. Default is 10."),
        type=int,
        default=10
    )
    parser.add_argument(
        '--min_dist',
        help=("The minimum distance apart that points are allowed to be in the low dimensional representation. Parameter is passed to the ParametricUMAP trainer. Default is 0.1."),
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--metric',
        help=("The name of the distance metric used during ParametricUMAP training. Default is 'euclidean'."),
        type=str,
        default="euclidean"
    )
    parser.add_argument(
        '--output_dir',
        help=("The output directory to save the trained model to. If None, the model name is used."),
        type=str,
        default=None
    ) 
    parser.add_argument(
        '--do_encoding',
        help=("Indicates that the dataset contains text data, and needs to be encoded before training the reduction. Default is to infer from the data."),
        default=None,
        action="store_true",
        dest="do_encoding"
    )
    parser.add_argument(
        '--no_encoding',
        help=("Indicates that the dataset already contains text encodings, so the data does not need to be encoded before training. Default is to infer from the data."),
        default=None,
        action="store_false",
        dest="do_encoding"
    )
    parser.add_argument(
        '--seed',
        help=("The seed to use for randomness in preprocessing. Default is None."),
        type=int,
        default=None
    )
    parser.add_argument(
        '--disable_tqdm',
        help=("Indicates that tqdm should be disabled. Default is False."),
        default=False,
        action="store_true"
    )
    parser.add_argument(
        '--verbose',
        '-v',
        help=("Whether to print preprocessing progress information. Default is False."),
        default=False,
        action="store_true"
    )

    kwargs = parser.parse_args()
    train(**vars(kwargs))