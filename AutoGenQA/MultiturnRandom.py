# %%
# ライブラリの自動リロード
from tqdm import tqdm
import random
import pandas as pd
from src.GGUFBot import GGUFBot
from datasets import load_dataset
import json
from datetime import datetime
import joblib
import random
import time

time.sleep(random.randint(0, 120))

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
random_number = random.randint(1, 10000)  # 1から100までのランダムな整数を生成
random_number_str = str(random_number)
out_path = f"data_random/completion_records_{random_number_str}_{current_time_no_symbols}.jsonl"

print("init model")
bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf",
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=5000)
print("fin initiating model")


noun_list=joblib.load("oasst_noun_list.bin")


while True:
    record={}
    record["database"]="random"
    nouns=random.choice(noun_list)
    nouns=random.sample(nouns,random.randint(1,len(nouns)))
    keywords=",".join(nouns)


    prompt = f"""キーワードをもとに､指示1にランダムな情報を入力し､UserとAssistantのやりとりを生成してください
キーワード: {keywords}
[テンプレート]
### 指示1:
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
