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


"""
covergence
Convergence occurs when the model’s performance stabilizes and the loss and accuracy curves level off. This is a sign that the model has learned the patterns in the data and is ready to be used. In simple terms, we are aiming for the model to converge to a stable performance every time we train it.

Once models have converged, we can use them to make predictions on new data and refer to evaluation metrics to understand how well the model is performing.

Intrepreting Learning Curves Patterns
1. Healthy Learning Curves
The loss can improve if the model’s output gets closer to the target, even if the final prediction is still incorrect. Accuracy, however, only improves when the prediction crosses the threshold to be correct.

Characteristics of healthy curves:

Smooth decline in loss: Both training and validation loss decrease steadily Close training/validation performance: Small gap between training and validation metrics Convergence: Curves level off, indicating the model has learned the patterns

During Trainig and After Training

During Training During the training process (after you’ve hit trainer.train()), you can monitor these key indicators:

Loss convergence: Is the loss still decreasing or has it plateaued?

Overfitting signs: Is validation loss starting to increase while training loss decreases?

Learning rate: Are the curves too erratic (LR too high) or too flat (LR too low)?

Stability: Are there sudden spikes or drops that indicate problems?

After Training After the training process is complete, you can analyze the complete curves to understand the model’s performance.

Final performance: Did the model reach acceptable performance levels?

Efficiency: Could the same performance be achieved with fewer epochs?

Generalization: How close are training and validation performance?

Trends: Would additional training likely improve performance?

"""

# overfitting

# it happens when the model learns too much from the training data and unable to generalize to different data 

"""
Symptoms:

Training loss continues to decrease while validation loss increases or plateaus.

Large gap between training and validation .

Training accuracy much higher than validation accuracy.

Solutions for overfitting:

Regularization: Add dropout, weight decay, or other regularization techniques

Early stopping: Stop training when validation performance stops improving

Data augmentation: Increase training data diversity

Reduce model complexity: Use a smaller model or fewer parameters

"""

# Example of detecting overfitting with early stopping
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    num_train_epochs=10,  # Set high, but we'll stop early
)

# Add early stopping to prevent overfitting
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)