# %%
from vllm import SamplingParams, LLM

from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import inst_dict, prepare_records

# バッチサイズ
n_records = 300


os.system("mkdir -p out_data")
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"out_data/model_{current_time_no_symbols}.jsonl"

ds_list = [
    # load_dataset("hpprc/jawiki-wiktionary", split="train"),
    load_dataset("hpprc/jawiki-books", split="train"),
    load_dataset("hpprc/wikipedia-20240101", split="train"),
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
model_name = "microsoft/Phi-3-medium-128k-instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
          )


# %%
try:
    ds = ds.shuffle()
except:
    pass

# %%


mode_list = list(inst_dict.keys())


# %%
print(len(ds), " records")

# %%
while True:
    records = prepare_records(
        ds, mode_list, n_records=n_records, random_extract=True)
    prompts = [record["original_text"] for record in records]
    outputs = llm.generate(
        prompts,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=1024,
            repetition_penalty=1.2,
        )
    )

    for record, output in zip(records, outputs):
        record["output_text"] = (output.outputs[0].text).strip()
        record.pop("original_text")
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
