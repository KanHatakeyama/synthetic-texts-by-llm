# %%
# %%
from vllm import SamplingParams, LLM
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import prepare_records
from src.clean_utils import clean
import random
import glob
import pandas as pd
import time

# %%

wait_time = random.randint(1, 10)
time.sleep(wait_time)

#####################
# 設定関連
n_records = 300
out_dir = "0624out_data_flan"

# parquet一覧
directory_path="/gs/bs/tga-hatakeyama/FLAN/"
# tubame
####################

################
# メイン

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}.jsonl"


# 再帰的にディレクトリ内のすべてのパーケットファイルを検索する
# parquet_list = glob.glob(f"{directory_path}/*/*.*.parquet", recursive=False)

# 再帰的にディレクトリ内のすべてのパーケットファイルを検索する
parquet_list = glob.glob(os.path.join(
    directory_path, '**', '*.parquet'), recursive=True)

#cotのみを使う
parquet_list = [file for file in parquet_list if 'cot' in os.path.dirname(file)]

# %%

def load_ds(parquet_list):

    while True:
        try:
            parquet_path = random.choice(parquet_list)
            # ファイルサイズを確認
            file_size = os.path.getsize(parquet_path)
            if file_size < 1000:
                continue
            print(f"load {parquet_path}")
            df = pd.read_parquet(parquet_path)
            ds = Dataset.from_pandas(df)
            break
        except Exception as e:
            print(e)
            time.sleep(10)
            continue


    print("shuffling")
    ds = ds.shuffle()

    return ds

ds=load_ds(parquet_list)

# %%


print("init llm")
model_name = "microsoft/Phi-3-medium-128k-instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
          )
inst_dict = {

    "translate1": """You are a professional translator. Translate the following Englsh into fluent Japanese.
#English""",
    "translate2": """You are a translator. Translate the following Englsh into fluent Japanese.
#English""",
    "translate3": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Use formal Japanese.
#English""",
    "translate4": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Use polite Japanese.
#English""",
    "translate5": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Use casual Japanese.
#English""",

}


mode_list = list(inst_dict.keys())


# %%


# %%
print(len(ds), " records")
while True:
    ds=load_ds(parquet_list)
    # 回答
    records = prepare_records(
        ds, mode_list, n_records=n_records,
        inst_dict=inst_dict

    )
    prompts = [record["original_text"] for record in records]
    outputs1 = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            repetition_penalty=1.15,
        )
    )

    for record, ja_output in zip(records, outputs1):
        ja = (ja_output.outputs[0].text).strip()

        ja = clean(ja, lang="ja")

        if ja == "":
            # print("rejected")
            # print(ja_output.outputs[0].text)
            continue

        record["ja"] = ja
        record.pop("original_text")

        # print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

# %%
