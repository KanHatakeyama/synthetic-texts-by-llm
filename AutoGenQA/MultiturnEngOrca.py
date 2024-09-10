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
time.sleep(random.randint(0, 120))

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
random_number = random.randint(1, 10000)  # 1から100までのランダムな整数を生成
random_number_str = str(random_number)
out_path = f"data_eng_orca/completion_records_{random_number_str}_{current_time_no_symbols}.jsonl"

print("init model")
bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf",
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=5000)
print("fin initiating model")


print("init original dataset")

# openorca
ds = load_dataset("Open-Orca/OpenOrca", split="train")
df = pd.DataFrame(ds)
df["database"] = "Open-Orca/OpenOrca_"+df["id"]
df = df.drop(columns=["system_prompt", "response", "id"], axis=1)
records = df.to_dict(orient='records')


while True:
    count = 0
    random.shuffle(records)
    for record in tqdm(records):
        eng_q = record["question"]
        prompt = f"""以下のテンプレートに従って､日本語のマルチターンの指示データを生成してください
・指示1,応答1､指示2､応答2が空欄になっているので､埋めてください
・会話の中身は､ランダムに決定してください
・テンプレートは厳守すること
・指示1は､次の英訳を､平易でわかりやすい日本語に意訳したものを使いなさい｡

[指示1の英訳]
{eng_q}
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
