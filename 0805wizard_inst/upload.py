"""
qrsh -g tga-hatakeyama -l cpu_4=1 -l h_rt=10:10:00
export HF_HOME="/gs/bs/tga-hatakeyama/hf_cache" 
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate synthtext
"""
import jsonlines
import pandas as pd
import pyarrow as pa
from huggingface_hub import HfApi, logging
import glob
from tqdm import tqdm
import os
import random
import json
from datetime import datetime

repo_id = "kanhatakeyama/wizardlm8x22b-logical-math-coding-sft"

logging.set_verbosity_debug()
hf = HfApi()

jsonl_dir_list = glob.glob(f"0805wizard8x22b/*.jsonl")

for integrated_file in jsonl_dir_list:
    with open(integrated_file, 'r') as file:
        print(f"Uploading {integrated_file}")
        hf.upload_file(
            path_or_fileobj=integrated_file,
            path_in_repo=f"data/{os.path.basename(integrated_file)}",
            repo_id=repo_id,
            repo_type="dataset"
        )
