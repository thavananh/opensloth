import os
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from evaluate import load

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

def compute_metrics(eval_pred):
    """Optional: a simple accuracy metric."""
    accuracy = load("accuracy")
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

def setup_model_and_training():
    # 1. Load a SMALL subset of IMDB for a quick test
    #    The 'split' argument selects a fraction, e.g. first 2k training, first 500 test
    raw_train = load_dataset("imdb", split="train[:2000]")
    raw_test  = load_dataset("imdb", split="test[:500]")

    # 2. Initialize a tokenizer + model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 3. Tokenize the dataset
    tokenized_train = raw_train.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    tokenized_test  = raw_test.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # 4. Define training arguments
    training_args = TrainingArguments(
        output_dir="test-hf-trainer",
        evaluation_strategy="epoch",
        logging_steps=50,
        save_strategy="epoch",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
    )

    # 5. Create a Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,    # Only needed if you plan to use Trainer for tokenization or generation
        compute_metrics=compute_metrics
    )

    return trainer
    # trainer.train()

    # 7. Evaluate
    # metrics = trainer.evaluate()
