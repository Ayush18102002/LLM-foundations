# In this we implement a training loop from scratch with modern PyTorch 
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""
Prepare for training
Before actually writing our training loop, we will need to define a few objects. The first ones are the dataloaders we will use to iterate over batches. But before we can define those dataloaders, we need to apply a bit of postprocessing to our tokenized_datasets, to take care of some things that the Trainer did for us automatically. Specifically, we need to:

Remove the columns corresponding to values the model does not expect (like the sentence1 and sentence2 columns).
Rename the column label to labels (because the model expects the argument to be named labels).
Set the format of the datasets so they return PyTorch tensors instead of lists.
Our tokenized_datasets has one method for each of those steps:

"""

tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")
tokenized_datasets["train"].column_names

# outputs - ['labels', 'input_ids', 'token_type_ids', 'attention_mask']

from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)

# To quickly check there is no mistake in the data processing, we can inspect a batch like this:
for batch in train_dataloader:
    break
{k: v.shape for k, v in batch.items()}


