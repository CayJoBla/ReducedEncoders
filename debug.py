"""
import torch
from sklearn.datasets import fetch_20newsgroups
from datasets import Dataset
from reduced_encoders import MPNetCompressedModel
from transformers import AutoTokenizer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# We will use the initial (untrained) all-mpnet-base-v2-compressed model
model_checkpoint = "cayjobla/all-mpnet-base-v2-compressed"
model = MPNetCompressedModel.from_pretrained(model_checkpoint, revision="initial").to(device)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the data
newsgroups = fetch_20newsgroups()
documents = Dataset.from_dict({"text":newsgroups.data, "target":newsgroups.target})

# Tokenize and get sentence embeddings in the data
def preprocess_data(batch):
    tokenized = tokenizer(batch["text"], truncation=True, padding="max_length", return_tensors="pt")
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    with torch.no_grad():
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = model.pooler(outputs[0], attention_mask) 
    return {"data": pooled_output.cpu().detach()}

embedding_dataset = documents.map(preprocess_data, batched=True, batch_size=250, remove_columns=documents.column_names).with_format("torch")
"""

import os
# os.environ["KERAS_BACKEND"] = "tensorflow"  # Use the tensorflow backend
os.environ["KERAS_BACKEND"] = "torch"       # Use the torch backend

import tensorflow as tf
print(tf.keras.backend.backend())

from umap.parametric_umap import ParametricUMAP
umap = ParametricUMAP(verbose=True)
umap.fit(tf.random.normal(20,768))