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

"""
{'labels': torch.Size([8]),
 'input_ids': torch.Size([8, 71]),
 'token_type_ids': torch.Size([8, 71]),
 'attention_mask': torch.Size([8, 71])}

"""


# create a model

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# test the model before training 
output = model(**batch)
print(output.loss, output.logits.shape)

# output = tensor(0.7710, grad_fn=<NllLossBackward0>) torch.Size([8, 2])-- 8 examples * 2 classes

# now comes the importtant part optimizer
from torch.optim import AdamW

optiizer = AdamW(
    model.parameters(), 
    lr=5e-5

)

# The thing that changes the model's weights to make the model better.

# what is lr = 5e-5
# lr means = learning ratee
# it controls how big the model's upadates are
# imagine you're walking toward the correct answer



from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
print(num_training_steps)

"""
output = 1337 steps

How many training steps?

Suppose:

Training examples = 3668
Batch size = 8

Approximately:

3668 / 8 ≈ 459 batches

If we train for 3 epochs:

459 × 3
≈ 1377 training steps

That's why the tutorial gets approximately:

1377

training steps.


"""

# choose cpua or gpu

import torch

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# training loop

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


