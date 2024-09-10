# %%
from vllm import SamplingParams, LLM
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import inst_dict, prepare_records
from src.clean_utils import clean
import random
import glob
import pandas as pd
import time
import sys

wait_time = random.randint(1, 10)
time.sleep(wait_time)

#####################
# 設定関連
n_records = 300
out_dir="0712out_data_ca"
directory_path="/gs/bs/tga-hatakeyama/TanukiCorpus/parquet_files"
####################

################
#メイン

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}.jsonl"

parquet_list=glob.glob(f"{directory_path}/*.parquet")

def extract_random_part(text):
    text_length = len(text)
    extract_length = min(text_length, random.randint(400, 2000))
    start_index = random.randint(0, text_length - extract_length)
    return text[start_index:start_index + extract_length]





def prepare_records(ds, mode_list,
                    random_extract=True,
                    n_records=300,
                    db_name="",
                    inst_dict=inst_dict,
                    ):
    ds = ds.shuffle()

    records = []
    cnt = 0
    for record in ds:
        #print(record)
        mode = random.choice(mode_list)
        inst = inst_dict[mode]

        # cosmopedia
        if "prompt" in record:
            key = random.choice(["prompt", "text"])
        else:
            key = "text"

        text = record[key]
        if random_extract:
            text = extract_random_part(text)
        text = f"""<|im_start|>user
{inst}{text}<|im_end|>
<|im_start|>assistant"""

        if "url" not in record:
            assert db_name != "", "url not found. you should set db_name"
            record["url"] = db_name
        records.append(
            {"original_text": text,
                "mode": mode,
                "url": record["url"]
             }
        )
        cnt += 1
        if cnt > n_records:
            break

    return records


while True:
 
    try:
        #if len(parquet_list)==0:
        #    break
        parquet_path=random.choice(parquet_list)
        #parquet_path="/storage5/shared/corpus/phase1_japanese/TanukiCorpus/parquet_files/10.parquet"
        print(f"load {parquet_path}")
        df=pd.read_parquet(parquet_path)
        ds= Dataset.from_pandas(df)
        break
    except Exception as e:
        print(e)
        time.sleep(10)
        continue

print("shuffling")
ds = ds.shuffle()

print("init llm")
model_name = "cyberagent/calm3-22b-chat"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=10000,
          #max_model_len=7000,
          #gpu_memory_utilization=0.4,
          )
inst_dict = {
"textbook": """以下のテキストから教科書を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"proof": """以下のテキストから校正された文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"logical": """以下のテキストから論理的な文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"Q&A": """以下のテキストからQ&Aを生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"conv": """以下のテキストから会話文を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
}              
inst_dict = {
"Q&A1": """次のテキストを書き直し、Q&Aを生成しなさい。見出しは出力しない\n#テキスト\n""",
"Q&A2": """次のテキストを書き直し、質疑応答を生成しなさい。見出しは出力しない\n#テキスト\n""",
}              

#ca系textのクリーン
def clean_ca_text(text):
    noise_keywords=[
"生成",
"質問と回答",
"テキスト",
    ]

    lines=text.split("\n")
    new_lines=[]
    for line in lines:
        for key in noise_keywords:
            found=False
            if line.find(key)>=0:
                found=True
                break
        if not found:
            new_lines.append(line)

    text="\n".join(new_lines)
    return text.strip()
       

mode_list = list(inst_dict.keys())


# %%
print(len(ds), " records")
while True:

    #スタイル変換
    records = prepare_records(
        ds, mode_list, n_records=n_records, random_extract=True,inst_dict=inst_dict,db_name="cc")
    prompts = [record["original_text"] for record in records]
    outputs1 = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            repetition_penalty=1.2,
        )
    )

    #英訳
    trans_prompts=[]
    for output in outputs1:
        ja_text=(output.outputs[0].text).strip()
        inst="Translate the following Japanese text into English."
        prompt = f"""<|im_start|>user
{inst}{ja_text}<|im_end|>
<|im_start|>assistant"""


        trans_prompts.append(prompt)
        #print(prompt)

    outputs2 = llm.generate(
        trans_prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            repetition_penalty=1.2,
        )
    )

    for record, ja_output,eng_output in zip(records, outputs1,outputs2):
        ja= (ja_output.outputs[0].text).strip()
        en=(eng_output.outputs[0].text).strip()

        ja=clean(ja,lang="ja")
        ja=clean_ca_text(ja) 
        en=clean(en,lang="en")

        if ja=="":
            #print("rejected")
            #print(ja_output.outputs[0].text)
            continue
        if en=="":
            #print("rejected")
            #print(eng_output.outputs[0].text)
            continue

        record["ja"]=ja 
        record["eng"] =en
        record["text"]=record["ja"]+"\n"+record["eng"]
        record.pop("original_text")


        #print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
