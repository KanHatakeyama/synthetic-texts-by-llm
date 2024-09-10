from tqdm import tqdm
import json
input_path="0723multiturn/upload/split_20240723_135913_0.jsonl"
output_path="0723multiturn/upload/filt.jsonl"

import json

def is_abnormal_text(text, threshold=10):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

# jsonlを読み込み､text keyを読み込み､is_abnormal_textで異常値を判定し､異常値を持つtextを除外
def filter_text(input_path, output_path):
    with open(output_path, 'w', encoding='utf-8') as f_out:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            for line in tqdm(f_in):
                data = json.loads(line)
                text = data['text']
                if not is_abnormal_text(text):
                    f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

#filter_text(input_path, output_path)

from huggingface_hub import HfApi, logging
import os
hf = HfApi()

repo_id = "kanhatakeyama/0723-calm3-22b-random-genre-inst-sft-multiturn-clean-tsub"

integrated_file = output_path
with open(integrated_file, 'r') as file:
    print(f"Uploading {integrated_file}")
    hf.upload_file(
        path_or_fileobj=integrated_file,
        path_in_repo=f"data/{os.path.basename(integrated_file)}",
        repo_id=repo_id,
        repo_type="dataset"
    )

