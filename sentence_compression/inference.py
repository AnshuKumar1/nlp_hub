import os
import yaml
import json
import pandas as pd

import evaluate
from transformers import pipeline


def load_pipeline(model_path):
    summarizer = pipeline("summarization", model=model_path, device=0)
    return summarizer

def infernece(pipeline, eval_data):
    prompt = "summarize the following sentence:"
    sentences = eval_data['original'].tolist()
    compressed = eval_data['compressed'].tolist()
    predictions = []
    for sent in sentences:
        text = prompt + sent
        out = pipeline(text)
        predictions.append(out[0]['summary_text'])
    return {"original": sentences, "compressed": compressed, "predictions": predictions}

def compute_performace(eval_data):
    original_compressed = eval_data['compressed']
    pred_compressed = eval_data['predictions']
    rouge = evaluate.load('rouge')
    predictions = eval_data['predictions']#.tolist()
    references = eval_data['compressed']#.tolist()
    # Compute the ROUGE score
    results = rouge.compute(predictions=predictions, references=references)
    print(results)
    return results

def get_latest_checkpoint(checkpoint_dir):
    subdirs = [name for name in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, name)) and name.startswith("checkpoint-")]
    checkpoint_numbers = [int(subdir.split("-")[1]) for subdir in subdirs]
    latest_checkpoint = "checkpoint-" + str(max(checkpoint_numbers))
    return latest_checkpoint

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    PROJECT_DIR = eval(config["SENTENCE_COMPRESSION"]["PROJECT_DIR"])
    data_dir = os.path.join(PROJECT_DIR, config["SENTENCE_COMPRESSION"]["DATA"]["CLEAN_DATA"])
    model_checkpoint = config["SENTENCE_COMPRESSION"]["INFERENCE"]["MODEL_PATH"]
    latest_checkpoint = get_latest_checkpoint(os.path.join(PROJECT_DIR, model_checkpoint))    
    model_path = os.path.join(PROJECT_DIR, model_checkpoint, latest_checkpoint)
    pipeline = load_pipeline(model_path)
    eval_data = pd.read_csv(os.path.join(data_dir, 'eval_data.csv'))
    eval_data_res = infernece(pipeline, eval_data)
    output_dir = os.path.join(PROJECT_DIR, config["SENTENCE_COMPRESSION"]["OUTPUT"]["RESULT"])
    os.makedirs(output_dir, exist_ok=True)
    eval_res_df = pd.DataFrame(eval_data_res)
    eval_res_df.to_csv(os.path.join(output_dir, "eval_result.csv"), index=False)
    result = compute_performace(eval_data_res)
    json.dump(result, open(os.path.join(output_dir, "performance.json"), "w"), indent=4)
