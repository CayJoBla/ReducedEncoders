# initialize_model.py

import sys
import argparse

from transformers import AutoTokenizer, BertTokenizer, BertConfig
from huggingface_hub import get_full_repo_name, Repository

from bert_reduced import BertReducedForMaskedLM, BertReducedForSequenceClassification


def main(base_model=None, reduced_size=48, task="fill-mask", model_name=None, tokenizer=None, output_dir=None, 
         commit_message=None, push_to_main=True, push_to_branch=True, branch_name="initial"):
    # Get model name, output directory, and full repository name
    if model_name is None:
        if base_model is None:
            model_name = "bert-reduced"
        else:
            model_name = f"{base_model}-reduced"
    output_dir = output_dir if output_dir else model_name
    repo_name = get_full_repo_name(model_name)

    # Load config
    if base_model:
        config = BertConfig.from_pretrained(base_model)
    config.reduced_size = reduced_size

    # Load tokenizer
    tokenizer = tokenizer if tokenizer else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer) if tokenizer else BertTokenizer()

    # Create the new model
    if task == "fill-mask":
        model = BertReducedForMaskedLM(config=config, _from_pretrained_base=base_model)
    elif task == "text-classification":
        model = BertReducedForSequenceClassification(config=config, _from_pretrained_base=base_model)
    else:
        raise ValueError("'task' parameter must be 'mlm' or 'text-classification'")
    
    # Get repository
    if push_to_main:
        repo = Repository(output_dir, clone_from=repo_name, revision="main")
    elif push_to_branch:
        repo = Repository(output_dir, clone_from=repo_name, revision=branch_name)
    
    # Save the model and tokenizer locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if not push_to_main and not push_to_branch: 
        return

    # Push model to the hub
    commit_message = commit_message if commit_message else \
        f"Initialize the reduced model with pretrained weights from the {base_model} base model"
    if push_to_main: 
        repo.push_to_hub(commit_message=commit_message, blocking=False)
    if push_to_branch:
        repo.git_checkout(branch_name)
        repo.push_to_hub(commit_message=commit_message, blocking=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Newly initialize a reduced BERT model and push to the model repository."
    )
    parser.add_argument(
        '--base_model',
        help=("The model to use as a base for the reduced BERT model. If not specified, all weights are randomly initialized."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--reduced_size',
        help=("The dimension of the output of the reduced model. Default is 48."),
        type=int,
        default=48
    )
    parser.add_argument(
        '--task',
        help=("The task that the new model is intended for ('fill-mask' or 'text-classification'). Default is 'fill-mask'."),
        type=str,
        default='fill-mask'
    )
    parser.add_argument(
        '--model_name',
        help=("The name of the model to use for the model repository. Defaults to '{base_model}-reduced'."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--tokenizer',
        help=("The tokenizer to use for this model. Defaults to the tokenizer of the base model."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--commit_message',
        help=("The commit message to use when pushing. If not specified, a default message is used."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--no_push_to_main',
        '-n',
        help=("Indicates that the model should not be pushed to the main branch. Default is False."),
        dest='push_to_main',
        default=True,
        action="store_false"
    )  
    parser.add_argument(
        '--push_to_branch',
        '-b',
        help=("Indicates that the model should be pushed to a non-main branch. Default is False."),
        default=False,
        action="store_true"
    )  
    parser.add_argument(
        '--branch_name',
        help=("The branch to push to if push_to_branch is True. Default is the 'initial' branch."),
        type=str,
        default="initial"
    )

    kwargs = parser.parse_args()

    main(**vars(kwargs))
