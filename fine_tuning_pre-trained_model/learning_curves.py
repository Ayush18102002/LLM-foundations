"""
learning curves 

The two curves

Loss curve – starts high (model knows nothing) and should smoothly go down as training continues. Lower = better.

Accuracy curve – starts low and should go up. But it moves in steps/plateaus, not smoothly. Why? Because accuracy only counts a prediction as "correct" once it crosses a threshold (like 0.5). A prediction can get closer to correct (loss improves) without actually flipping to correct yet — so accuracy doesn't move until it crosses that line.

Think of it like: loss is like grading a paper with partial credit, and accuracy is like a pass/fail exam — you can be improving toward passing without your pass/fail status changing.




"""

# loss curves 
# it shows the model error decreases over time. in a typical sucessful training run 

"""
High initial loss: The model starts without optimization, so predictions are initially poor
Decreasing loss: As training progresses, the loss should generally decrease
Convergence: Eventually, the loss stabilizes at a low value, indicating that the model has learned the patterns in the data

"""

# Example of tracking loss during training with the Trainer
from transformers import Trainer, TrainingArguments
import wandb

# Initialize Weights & Biases for experiment tracking
wandb.init(project="transformer-fine-tuning", name="bert-mrpc-analysis")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    logging_steps=10,  # Log metrics every 10 steps
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    report_to="wandb",  # Send logs to Weights & Biases
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# Train and automatically log metrics
trainer.train()