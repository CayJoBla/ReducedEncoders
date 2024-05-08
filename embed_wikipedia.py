# embed_wikipedia.py

import torch
from transformers import AutoTokenizer, set_seed
from reduced_encoders import MPNetCompressedModel, MPNetReducedConfig
from datasets import load_dataset


def embed_wikipedia():
    # Load config, tokenizer, and model
    model_name = "cayjobla/all-mpnet-base-v2-compressed"
    revision = "initial"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = MPNetReducedConfig.from_pretrained(model_name, revision=revision)
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = MPNetCompressedModel.from_pretrained(model_name, config=config, revision=revision)
    model.to(device)

    # Load the wikipedia dataset
    dataset = load_dataset("wikipedia" , "20220301.en", split="train")

    # Tokenize the dataset
    def tokenize_data(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

    tokenized = dataset.map(tokenize_data, batched=True, batch_size=1000, remove_columns=dataset.column_names)

    # Embed the dataset
    def embed_data(batch):
        input_ids = torch.tensor(batch["input_ids"]).to(device)
        attention_mask = torch.tensor(batch["attention_mask"]).to(device)
        with torch.no_grad():
            outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = model.pooler(outputs[0], attention_mask) 
        return {"data": pooled_output.cpu().detach()}

    embedded = tokenized.map(embed_data, batched=True, batch_size=100, remove_columns=tokenized.column_names)
    embedding_dataset = embedded.with_format("torch")

    # Push to Huggingface
    embedding_dataset.push_to_hub("cayjobla/wikipedia_embedded")

if __name__ == "__main__":
    set_seed(916)
    embed_wikipedia()