# %%
# ライブラリの自動リロード
from tqdm import tqdm
import random
import pandas as pd
from src.GGUFBot import GGUFBot
from datasets import load_dataset
import json
from datetime import datetime
import time

time.sleep(random.randint(0, 15))

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"data_multi_misc/completion_records{current_time_no_symbols}.jsonl"




print("init original dataset")


records=[]
q_list=[]

#tsubameのネットワークエラーが出るので､try exceptで回避

try:
    #oasst
    ds=load_dataset("llm-jp/oasst1-21k-ja",)["train"]
    for record in ds:
        conversations=record["conversations"]
        q=conversations[0]["value"]
        if q not in q_list:
            q_list.append(q)
            records.append(
                {"question":q,
                "database":"llm-jp/oasst2-33k-ja"},
            )
except Exception as e:
    print(e)

#minnade
try:
    ds=load_dataset("minnade/chat-daily",)["train"]
    for record in ds:
        q=record["body"]
        if q=="":
            continue
        if record["role"]!="user":
            continue
        if q not in q_list:
            q_list.append(q)
            records.append(
                {"question":q,
                "database":"minnade/chat-daily"},
            )
except Exception as e:
    print(e)

#dolly
try:
    ds=load_dataset("kunishou/databricks-dolly-15k-ja",)["train"]
    for record in ds:
        q=record["instruction"]
        if q=="":
            continue
        inp=record["input"]
        if inp!="":
            q=f"{q}\n\n{inp}"

        if q not in q_list:
            q_list.append(q)
            records.append(
                {"question":q,
                "database":"kunishou/databricks-dolly-15k-ja"},
            )
except Exception as e:
    print(e)

#chatbotarena
try:

    ds=load_dataset("cyberagent/chatbot-arena-ja-calm2-7b-chat-experimental",split="train")
    for record in ds:
        q=record["prompt"]
        if q=="":
            continue

        if q not in q_list:
            q_list.append(q)
            records.append(
                {"question":q,
                "database":"cyberagent/chatbot-arena-ja-calm2-7b-chat-experimental"},
            )
except Exception as e:
    print(e)

print("init model")
bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf",
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=500)
print("fin initiating model")

while True:
    count = 0
    random.shuffle(records)
    for record in tqdm(records):
        q = record["question"]
        prompt = f"""以下のテンプレートに従って､日本語のマルチターンの指示データを生成してください
・応答1､指示2､応答2が空欄になっているので､埋めてください
・会話の中身は､ランダムに決定してください
・テンプレートは厳守すること

[テンプレート]
### 指示1:{q}
### 応答1:
### 指示2:
### 応答2:
"""
        try:
            a = bot.ask(prompt)
            if a == "":
                continue
            record["autogen_text"] = a
        except Exception as e:
            print(record, e)
            continue

        print(record)
        with open(out_path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')
