import argparse, torch, evaluate
import os

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(sample):
    # tokenize BERT
    return tokenizer(sample["sentence1"], sample["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # use dynamic padding

metric = evaluate.load("glue", "mrpc")

def compute_metrics(eval_pred):
    # make predictions and calculate accuracy
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def main():
    parser = argparse.ArgumentParser(description="BERT fine-tuning")

    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default="./results")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = main()

    train_dataset = tokenized_datasets["train"]
    if args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(args.max_train_samples))

    eval_dataset = tokenized_datasets["validation"]
    if args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    training_args = TrainingArguments(
        output_dir=args.model_path,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        report_to="wandb",
        eval_strategy="epoch",
        save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if args.do_train:
        # train and save model
        print(f"--- Launching Training Loop (LR={args.lr}, Batch={args.batch_size}) ---")
        trainer.train()
        trainer.save_model(args.model_path)

    if args.do_predict:
        if os.path.exists(args.model_path):
            # load trained model
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
            model = model.to(trainer.args.device)
            trainer.model = model

        test_dataset = tokenized_datasets["test"]
        raw_test_dataset = raw_datasets["test"]

        if args.max_eval_samples != -1:
            test_dataset = test_dataset.select(range(args.max_eval_samples))

        model.eval()
        predictions = trainer.predict(test_dataset)
        raw_logits = predictions.predictions
        predictions = np.argmax(raw_logits, axis=-1)

        # save predictions
        os.makedirs(args.model_path, exist_ok=True)
        with open(os.path.join(args.model_path, "predictions.txt"), "w") as f:
            for item, pred in zip(raw_test_dataset, predictions):
                sent1 = item["sentence1"]
                sent2 = item["sentence2"]
                f.write(f"{sent1}###{sent2}###{pred}\n")