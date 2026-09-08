from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate
import numpy as np


# 1. Dataset
raw_datasets = load_dataset("glue", "mrpc")


# 2. Model name
checkpoint = "bert-base-uncased"


# 3. Tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# 4. Tokenize
def tokenize_function(example):
    return tokenizer(
        example["sentence1"],
        example["sentence2"],
        truncation=True
    )

tokenized_datasets = raw_datasets.map(
    tokenize_function,
    batched=True
)


# 5. Model
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint,
    num_labels=2
)


# 6. Training settings
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch"
)


# 7. Metrics
metric = evaluate.load("glue", "mrpc")

def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(
        predictions=predictions,
        references=labels
    )


# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


# 9. Train!
trainer.train()