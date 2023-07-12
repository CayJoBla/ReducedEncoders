from huggingface_hub import get_full_repo_name, Repository
import argparse

def main(local_dir=None, branch='main', verify=False, commit_message=None, push=True):
    local_dir = local_dir if local_dir else "bert-base-uncased-reduced"
    repo = Repository(local_dir, clone_from=get_full_repo_name(local_dir)) 
    repo.git_checkout(branch)

    if verify:
        confirmation = input(f"Is '{repo.current_branch}' the branch you wish to push to? (y)/n: ")
        if confirmation.lower() == "n":
            print("Cancelling push...")
            return 1
        
    repo.git_add()

    commit_message = commit_message if commit_message else "Save model"
    
    repo.git_commit(commit_message=commit_message)      # Handle errors

    repo.git_push(f'origin {branch}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push reduced BERT model to huggingface hub.")
    parser.add_argument(
        '--local_dir',
        '-d',
        help=("The path to the local git repository containing the model. Also the name of the model being pushed to. "
              "Default is bert-base-uncased-reduced."),
        type=str,
        default=None
    )
    parser.add_argument(
        '--branch',
        '-b',
        help=("The branch to push to. Default is 'main'."),
        type=str,
        default="main"
    )
    parser.add_argument(
        '--verify',
        '-v',
        help=("Verify with the user before committing. Default is False."),
        default=False,
        action="store_true"
    )  
    parser.add_argument(
        '--commit_message', 
        '-m',
        help=("The commit message to use. Default is 'Save model'"),
        type=str,
        default=None
    )
    parser.add_argument(
        '--no_push',
        help=("Indicates that the script should only commit and not push the changes. Default is False."),
        dest='push',
        default=True,
        action="store_false"
    )

    kwargs = parser.parse_args()

    main(**vars(kwargs))