# %%
# %%
from llama_cpp import Llama
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
n_records = 5
out_dir = "0626out_data_qa"


# tubame
####################

################
# メイン

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}.jsonl"


# %%
# textフィールドとoutputフィールドを結合する関数を定義
def concatenate_text_output(example):
    example["text"] = example["text"] + "\n" + example["output"]
    return example


code_dataset = load_dataset("flytech/python-codes-25k", split="train")

# データセット全体に適用
modified_dataset = code_dataset.map(concatenate_text_output)

# %%
ds_list = [
    load_dataset("hpprc/jawiki-books", split="train"),
    load_dataset("hpprc/wikipedia-20240101", split="train"),
    load_dataset("geniacllm/hanrei_v2", split="train"),
    code_dataset,
]
# text と url 列だけを抽出して新しいリストに追加
ds_list_filtered = [
    ds.remove_columns(
        [col for col in ds.column_names if col not in ['text', 'url']])
    for ds in ds_list
]

# データセットを結合
ds = concatenate_datasets(ds_list_filtered)
# ds = concatenate_datasets(ds_list)


try:
    ds = ds.shuffle()
except:
    pass

# %%
model_path="/gs/fs/tga-hatakeyama/model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf"

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

        prompt = f"""<s>[INST]{question}[/INST]"""

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

    "qa1": """Generate questions and answers based on the following #text.
-Include the information in #text in your questions.
-Output only questions and answers.
-Output only Japanese sentence.
-Respondents cannot see the #text, so any information necessary for the answer MUST BE included in the question itself.
#text""",

    "qa2": """Generate difficult questions and answers based on the following #text.
-Include the information in #text in your questions.
-Output only questions and answers.
-Output only Japanese sentence.
-Respondents cannot see the #text, so any information necessary for the answer MUST BE included in the question itself.
#text""",

    "qa3": """Generate elementary school level questions and answers based on the following #text.
-Include the information in #text in your questions.
-Output only questions and answers.
-Output only Japanese sentence.
-Respondents cannot see the #text, so any information necessary for the answer MUST BE included in the question itself.
#text""",

    "qa3": """Generate high school level questions and answers based on the following #text.
-Include the information in #text in your questions.
-Output only questions and answers.
-Output only Japanese sentence.
-Respondents cannot see the #text, so any information necessary for the answer MUST BE included in the question itself.
#text""",

    "qa4": """Generate university level questions and answers based on the following #text.
-Include the information in #text in your questions.
-Output only questions and answers.
-Output only Japanese sentence.
-Respondents cannot see the #text, so any information necessary for the answer MUST BE included in the question itself.
#text""",

    "qa4": """Generate questions and answers based on the following #text.
-Include the information in #text in your questions.
-Output only questions and answers.
-Output only Japanese sentence.
-Respondents cannot see the #text, so any information necessary for the answer MUST BE included in the question itself.
-Output Q&A as json data
#text""",

}


mode_list = list(inst_dict.keys())


# %%
# bot.ask("元気ですか?")

# %%
def random_excerpt(text, max_length=2000):
    if len(text) <= max_length:
        return text
    else:
        start_index = random.randint(0, len(text) - max_length)
        return text[start_index:start_index + max_length]


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

        text = record["text"]

        if len(text) > 2000:
            text = random_excerpt(text, max_length=2000)

        prompt = f"""{inst}{text}"""

        records.append(
            {"prompt": prompt,
                "mode": mode,
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
        try:
            outputs1.append(bot.ask(prompt))
        except Exception as e:
            print(e)
            time.sleep(10)
            continue

    for record, ja_output in zip(records, outputs1):
        ja = (ja_output).strip()
        # ja=clean(ja,lang="ja")

        if ja == "":
            # print("rejected")
            # print(ja_output.outputs[0].text)
            continue

        record["text"] = ja
        record.pop("prompt")

        # print("saving to "+out_path)
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
