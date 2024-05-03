import os
import yaml
import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, Trainer
from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np


checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize the following sentence: "
def preprocess_function(examples):
    inputs = prefix + examples["original"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    labels = tokenizer(text_target=examples["compressed"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge = evaluate.load("rouge")
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    return {k: round(v, 4) for k, v in result.items()}

def main():
    print("Data Loading...")
    config = yaml.safe_load(open("config.yaml", "r"))
    PROJECT_DIR = eval(config["SENTENCE_COMPRESSION"]["PROJECT_DIR"])
    data_dir = os.path.join(PROJECT_DIR, config["SENTENCE_COMPRESSION"]["DATA"]["CLEAN_DATA"])
    data = pd.read_csv(os.path.join(data_dir, 'training_data.csv'))
    print("Tokenization started...")
    data_preprocessed = data.apply(preprocess_function, axis=1)
    print("Test data preprocessing...")
    train_tokenized, test_tokenized = train_test_split(data_preprocessed, test_size=0.2)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
    print("Model Loading...")
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    training_args = Seq2SeqTrainingArguments(
        output_dir="checkpoints",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized.values,
        eval_dataset=test_tokenized.values,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    
if __name__ == "__main__":
    main()
        