# %%
from vllm import SamplingParams, LLM
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from datasets import Dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import inst_dict, prepare_records
import random
import glob
import pandas as pd
import time
wait_time = random.randint(1, 60)
time.sleep(wait_time)


# バッチサイズ
n_records = 300
#n_records = 30


os.system("mkdir -p out_data_cc")
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"out_data_cc/model_{current_time_no_symbols}.jsonl"

directory_path="/gs/bs/tga-hatakeyama/TanukiCorpus/parquet_files"
parquet_list=glob.glob(f"{directory_path}/*.parquet")

while True:
    try:
        parquet_path=random.choice(parquet_list)
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
    "logical": """以下のテキストから論理的な文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "Q&A": """以下のテキストからQ&Aを生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "essay": """以下のテキストから随筆を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "news_article": """以下のテキストをニュース記事に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "report": """以下のテキストをレポートに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "story": """以下のテキストを物語に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "summary": """以下のテキストを要約しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "letter": """以下のテキストを手紙に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "review": """以下のテキストをレビューに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",    "interview": """以下のテキストをインタビュー形式に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テ>キスト\n""",
    "instructions": """以下のテキストを指示に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "blog_post": """以下のテキストをブログ記事に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "speech": """以下のテキストをスピーチに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",    "manual": """以下のテキストをマニュアルに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",                                                                                                                                              "guide": """以下のテキストをガイドに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "presentation": """以下のテキストをプレゼンテーションに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",                                                                                                                                "advertisement": """以下のテキストを広告に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",                                                                                                                                             "announcement": """以下のテキストを発表文に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",                                                                                                                                            "fiction": """以下のテキストをフィクションに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",                                                                                                                                           "nonfiction": """以下のテキストをノンフィクションに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",                                                                                                                                    "dialogue": """以下のテキストを対話形式に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
}
               
inst_dict = {
"textbook": "以下のテキストを教科書に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n",
"kosei": "以下のテキストを校正して書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n",
"logical": "以下のテキストから論理的な文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n",
"Q&A": "以下のテキストからQ&Aを生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n",
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
            repetition_penalty=1.2,
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
            repetition_penalty=1.2,
        )
    )

    for record, ja_output,eng_output in zip(records, outputs1,outputs2):
        record["ja"] = (ja_output.outputs[0].text).strip()
        record["eng"] = (eng_output.outputs[0].text).strip()
        record.pop("original_text")
        #print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

