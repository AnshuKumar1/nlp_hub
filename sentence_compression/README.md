## Getting Started

### Installation

1. conda environment
`conda env create --name NAME --file=environment.yaml`


The Assignment is designed around several scripts that simulate a typical machine learning workflow. Starting with data preparation after preparing data, training model, evaluation and inference model. `google/t5-small` model was being trained on above dataset for `10` epochs. Later inference ran on evaluation data, performance metrics and evaluation results were stored inside `result` subdirectory of `project` directory.

I added Makefile which can be used to run python scripts separately using following bash commands.

```bash
make data
make train
make eval
make inference
```

`run` is a bash command which can aggregately run entire project.

```bash
make run
```

`clean` is a bash command which can be used to clean the previous runs.

```bash
make clean
```

Performance metrics stores into `performance.json` file inside `results` directory.

```json
{
    "rouge1": 0.79689240266461,
    "rouge2": 0.7606140631154827,
    "rougeL": 0.7733855633904199,
    "rougeLsum": 0.7734703253159519
}
```

And also, `eval_results.csv` containing predictions of evaluation file.

| original  | compressed | predictions |
|-----------|------------|-------------|
| sentence1 | compress1  | prediction1 |
| sentence2 | compress2  | prediction2 |
| :         | :          | :           |

### References:
1. https://github.com/google-research-datasets/sentence-compression 
2. https://huggingface.co/docs/transformers/en/tasks/summarization 

### Note:
Download trained checkpoint from given drive link [checkpoint](https://drive.google.com/drive/folders/1yrl0VtmM9BtT4aU2Z5vLs6doz35MMxvM?usp=drive_link)


