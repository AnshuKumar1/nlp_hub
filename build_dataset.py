import os
import shutil
import glob
import json
import csv
import yaml
from git import Repo 
import gzip


fieldnames = ['original','compressed']

def to_csv_record(writer, buffer):
  record = json.loads(buffer)
  writer.writerow(dict(
    original=record['graph']['sentence'],
    compressed=record['compression']['text']))

def build_dataset(rawdata_dir, preprocessed_data_dir):
    print("Data Preparation...")
    os.makedirs(preprocessed_data_dir, exist_ok=True)
    with open(os.path.join(preprocessed_data_dir, 'training_data.csv'),'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      writer.writeheader()
      for rawdata_files in glob.glob(f'{rawdata_dir}/data/**train**.json'):
        with open(rawdata_files) as raw_contents:
          buffer = ''
          for line in raw_contents:
            if line.strip()=='':
                to_csv_record(writer, buffer)
                buffer = ''
            else:
              buffer += line
          if len(buffer)>0: 
            to_csv_record(writer, buffer)

    with open(os.path.join(preprocessed_data_dir, 'eval_data.csv'),'w') as csvfile:
      writer = csv.DictWriter(csvfile, fieldnames=['original','compressed'])
      writer.writeheader()
      with open(f'{rawdata_dir}/data/comp-data.eval.json') as raw_contents:
        buffer = ''
        for line in raw_contents:
          if line.strip()=='':
              to_csv_record(writer, buffer)
              buffer = ''
          else: buffer += line
        if len(buffer)>0: to_csv_record(writer, buffer)
    
def decompressing_rawdata(rawdata_dir):
    print("Decompression...")
    compressed_files = glob.glob(rawdata_dir + "/data/*.json.gz")
    for compressed_file_path in compressed_files:
        output_file_path = os.path.splitext(compressed_file_path)[0] 
        with gzip.open(compressed_file_path, 'rb') as comp_file:
            compressed_content = comp_file.read()
        with open(output_file_path, 'wb') as output_file:
            output_file.write(compressed_content)
        os.remove(compressed_file_path)

def download_rawdata(git_url, rawdata_dir):
    os.makedirs(rawdata_dir, exist_ok=True)
    print("Data Cloning...")
    current_dir = os.getcwd()
    try:
        os.chdir(rawdata_dir)
        Repo.clone_from(git_url, '.')
    except Exception as e:
        print("Error:", e)
    finally:
        os.chdir(current_dir)
    
if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml", "r"))
    PROJECT_DIR = eval(config["SENTENCE_COMPRESSION"]["PROJECT_DIR"])
    rawdata_git = config["SENTENCE_COMPRESSION"]["DATA"]["RAW_DATA"]
    preprocessed_data_dir = os.path.join(PROJECT_DIR, config["SENTENCE_COMPRESSION"]["DATA"]["CLEAN_DATA"])
    rawdata_dir = os.path.join(PROJECT_DIR, config["SENTENCE_COMPRESSION"]["DATA"]["RAW_DIR"])
    download_rawdata(rawdata_git, rawdata_dir)
    decompressing_rawdata(rawdata_dir)
    build_dataset(rawdata_dir, preprocessed_data_dir)
    shutil.rmtree(rawdata_dir)
