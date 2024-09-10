"""
export HF_HOME="/gs/bs/tga-hatakeyama/hf_cache" 
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate synthtext
cd /gs/bs/tga-hatakeyama/ca_text/
"""
import pandas as pd
import pyarrow as pa
from huggingface_hub import HfApi, logging
import glob
from tqdm import tqdm
import os
import random
import json
from datetime import datetime

repo_id = "kanhatakeyama/CreativeCommons-RAG-QA-Calm3-22b-chat"
jsonl_dir = "0712out_data_ca/"
jsonl_list = glob.glob(f"{jsonl_dir}/*.jsonl")
jsonl_list.sort()

logging.set_verbosity_debug()
hf = HfApi()


import os
import jsonlines


def get_file_size(file_path):
    return os.path.getsize(file_path)

def load_processed_files(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_processed_files(file_path, processed_files):
    with open(file_path, 'w') as file:
        json.dump(processed_files, file)
def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")
def split_jsonl_files(input_files, output_dir, processed_file_path, max_size_in_gb=10):
    max_size = max_size_in_gb * 1024 * 1024 * 1024  # Convert GB to bytes
    current_size = 0
    file_index = 0
    writer = None

    os.makedirs(output_dir, exist_ok=True)
    
    processed_files = load_processed_files(processed_file_path)
    
    for input_file in tqdm(input_files):
        if input_file in processed_files:
            print(f"Skipping already processed file: {input_file}")
            continue
        
        with jsonlines.open(input_file, 'r') as reader:
            for obj in reader:
                if writer is None or current_size >= max_size:
                    if writer is not None:
                        writer.close()
                    #output_file = os.path.join(output_dir, f"split_{file_index}.jsonl")
                    timestamp = get_timestamp()
                    output_file = os.path.join(output_dir, f"split_{timestamp}_{file_index}.jsonl")
                    writer = jsonlines.open(output_file, 'w')
                    current_size = 0
                    file_index += 1
                
                writer.write(obj)
                current_size += len(json.dumps(obj).encode('utf-8'))
        
        processed_files.append(input_file)
        save_processed_files(processed_file_path, processed_files)
    
    if writer is not None:
        writer.close()

# 使用例
input_files=jsonl_list  # 入力ファイルのリストを指定
output_dir = f"{jsonl_dir}/upload/"  # 出力ディレクトリのパスを指定
processed_file_path = f"{jsonl_dir}/upload/processed_files.json"  # 処理済みファイルのリストを保存するJSONファイルのパス
split_jsonl_files(input_files, output_dir, processed_file_path)



for integrated_file in glob.glob(f"{jsonl_dir}/upload/*.jsonl"):
    with open(integrated_file, 'r') as file:
        print(f"Uploading {integrated_file}")
        hf.upload_file(
            path_or_fileobj=integrated_file,
            path_in_repo=f"data/{os.path.basename(integrated_file)}",
            repo_id=repo_id,
            repo_type="dataset"
        )

