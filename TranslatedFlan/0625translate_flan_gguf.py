# %%
# %%
from llama_cpp import Llama
from datasets import Dataset
from datetime import datetime
import json
import os
from src.generator import prepare_records
import random
import glob
import pandas as pd
import time

# %%

wait_time = random.randint(1, 10)
time.sleep(wait_time)

#####################
# 設定関連
n_records = 30
out_dir = "0625out_data_flan_mixtral"

# parquet一覧

directory_path="/gs/bs/tga-hatakeyama/FLAN/"
# tubame
####################

# %%
#model_path = "/data/2023/1505llmmatsu/mixtral_gguf/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf"

#tsubame
model_path="/gs/fs/tga-hatakeyama/model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf"

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

#parquet_list = [i for i in parquet_list if i.find("cot") > 0]
print(len(parquet_list))

# %%

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
        # time.sleep(10)
        continue


print("shuffling")
ds = ds.shuffle()


class GGUFBot:
    def __init__(self, model_path="model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf",
                 max_new_tokens=4000,
                 n_gpu_layers=100,
                 n_ctx=4096) -> None:
        print("loading model...")

        self.model = Llama(model_path=model_path,
                           n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, )
        self.max_new_tokens = max_new_tokens

    def ask(self, question):

        prompt = f"""<s>[INST]{question}[/INST]#日本語訳\n"""

        output = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            # temperature = 0.7,
            # top_p = 0.8,
            # repeat_penalty = 1.1,
            # frequency_penalty = 1.0,
            # presence_penalty = 1.0,
            # stop = ["\n###  Instruction:", "\n### Response:", "\n"],
            # echo = True,
        )
        return output["choices"][0]["text"].strip()


bot = GGUFBot(model_path=model_path)


# %%


print("init llm")
# model_name = "microsoft/Phi-3-medium-128k-instruct"
# llm = LLM(model=model_name, trust_remote_code=True,
#          max_model_len=20000
#          )
inst_dict = {

    "translate1": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Output only the translated Japanese sentence.
#English
He is a good man.
#日本語訳
彼はいい人です。
#English
""",

    "translate2": """You are a translator. Translate the following Englsh into fluent Japanese.
Output only the translated Japanese sentence.
#English
He is a good man.
#日本語訳
彼はいい人です。
#English
""",

    "translate3": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Use formal Japanese.
Output only the translated Japanese sentence.
#English
He is a good man.
#日本語訳
彼はいい人です。
#English
""",

    "translate4": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Use polite Japanese.
Output only the translated Japanese sentence.
#English
He is a good man.
#日本語訳
彼はいい人です。
#English
""",

    "translate5": """You are a professional translator. Translate the following Englsh into fluent Japanese.
Use casual Japanese.
Output only the translated Japanese sentence.
#English
He is a good man.
#日本語訳
彼はいい人です。
#English
""",

}


mode_list = list(inst_dict.keys())


# %%
# bot.ask("元気ですか?")

# %%
def prepare_records(ds, mode_list,
                    inst_dict,
                    n_records=300,
                    ):
    ds = ds.shuffle()

    records = []
    cnt = 0
    for record in ds:
        mode = random.choice(mode_list)
        inst = inst_dict[mode]

        text = record["inputs"]+"\n"+record["targets"]
        prompt = f"""{inst}{text}"""

        records.append(
            {"prompt": prompt,
                "mode": mode,
                "en": text
             }
        )
        cnt += 1
        if cnt > n_records:
            break

    return records


# %%

# %%
print(len(ds), " records")
while True:

    # 回答
    records = prepare_records(
        ds, mode_list, n_records=n_records,
        inst_dict=inst_dict

    )
    prompts = [record["prompt"] for record in records]

    outputs1 = []
    for prompt in prompts:
        outputs1.append(bot.ask(prompt))

    for record, ja_output in zip(records, outputs1):
        ja = (ja_output).strip()
        # ja=clean(ja,lang="ja")

        if ja == "":
            # print("rejected")
            # print(ja_output.outputs[0].text)
            continue

        record["ja"] = ja
        record.pop("prompt")

        # print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
