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
wait_time = random.randint(1, 10)
time.sleep(wait_time)

#####################
# 設定関連
n_records = 300
out_dir="0617out_data_cleaned"
directory_path="/storage5/shared/corpus/phase1_japanese/TanukiCorpus/parquet_files"

#tubame
directory_path="/gs/bs/tga-hatakeyama/TanukiCorpus/parquet_files"
####################

################
#メイン

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}.jsonl"

parquet_list=glob.glob(f"{directory_path}/*.parquet")

while True:
    try:
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
model_name = "microsoft/Phi-3-medium-128k-instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
          )
inst_dict = {                                                                                                                                   "textbook": """以下のテキストを教科書に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",    "conversation": """以下のテキストから会話文を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"textbook": """以下のテキストから教科書を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"proof": """以下のテキストから校正された文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"logical": """以下のテキストから論理的な文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
"Q&A": """以下のテキストからQ&Aを生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
}              

       

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
            repetition_penalty=1.15,
        )
    )

    #英訳
    trans_prompts=[]
    for output in outputs1:
        ja_text=(output.outputs[0].text).strip()
        inst="Translate the following Japanese text into English."
        prompt= f"""<|user|>
{inst}{ja_text}<|end|>
<|assistant|>"""
        trans_prompts.append(prompt)
        #print(prompt)

    outputs2 = llm.generate(
        trans_prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=2048,
            repetition_penalty=1.15,
        )
    )

    for record, ja_output,eng_output in zip(records, outputs1,outputs2):
        ja= (ja_output.outputs[0].text).strip()
        en=(eng_output.outputs[0].text).strip()

        ja=clean(ja,lang="ja")
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
        record.pop("original_text")


        #print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
