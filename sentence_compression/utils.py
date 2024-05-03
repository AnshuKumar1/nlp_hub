import os
import json
import pandas as pd
import yaml
import seaborn as sns
import matplotlib.pyplot as plt
from inference import get_latest_checkpoint


def process_loss(loss, final_loss):
    epoch = int(loss["epoch"])
    final_loss["epoch"].append(epoch)
    for key in ["loss", "eval_loss", "eval_rouge1", "eval_rouge2"]:
        try:
            value = loss[key]
            final_loss[key].append(value)
        except KeyError:
            pass

def loss_function(losses):
    final_loss = {
        "epoch": [],
        "loss": [],
        "eval_loss": [],
        "eval_rouge1": [],
        "eval_rouge2": []
    }
    for loss_steps in losses:
        if float(loss_steps.get("epoch", 0)) % 1 == 0:
            process_loss(loss_steps, final_loss)
    final_loss["epoch"] = list(set(final_loss["epoch"]))
    return final_loss

def plot_loss(data, output_dir):
    df = pd.DataFrame(data)
    df_melted = pd.melt(df, id_vars=['epoch'], var_name='metric', value_name='value')
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x='epoch', y='value', hue='metric', marker='o')
    plt.legend(title='Metric')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Metrics vs Epoch')
    plt.savefig(os.path.join(output_dir, 'metrics_vs_epoch.png'))


if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    PROJECT_DIR = eval(config["SENTENCE_COMPRESSION"]["PROJECT_DIR"])
    checkpoint_dir = config["SENTENCE_COMPRESSION"]["INFERENCE"]["MODEL_PATH"]
    latest_checkpoint = get_latest_checkpoint(os.path.join(PROJECT_DIR, checkpoint_dir))
    logfile_dir = os.path.join(PROJECT_DIR, checkpoint_dir, latest_checkpoint)
    logfile_path = os.path.join(logfile_dir, "trainer_state.json")
    logs = json.load(open(logfile_path))
    final_loss = loss_function(logs["log_history"])
    output_dir =  config["SENTENCE_COMPRESSION"]["OUTPUT"]["RESULT"]
    os.makedirs(output_dir, exist_ok=True)
    plot_loss(final_loss, output_dir)