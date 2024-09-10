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


# %%

# ds=load_dataset("kanhatakeyama/ChatbotArenaJaMixtral8x22b", split="train")
# ds = load_dataset("wikipedia", "20220301.en", streaming=False, split="train")
# ds=load_dataset("hpprc/jawiki-wiktionary", split="train")
# ds = load_dataset("hpprc/jawiki-books", split="train")
streaming = False
ds_list = [
    load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text",
                 streaming=streaming, split="train"),
    load_dataset("HuggingFaceTB/cosmopedia", "khanacademy",
                 streaming=streaming, split="train"),
    load_dataset("HuggingFaceTB/cosmopedia", "openstax",
                 streaming=streaming, split="train"),
    load_dataset("HuggingFaceTB/cosmopedia", "stanford",
                 streaming=streaming, split="train"),
    load_dataset("HuggingFaceTB/cosmopedia", "wikihow",
                 streaming=streaming, split="train"),
]
ds = concatenate_datasets(ds_list)

# %%
try:
    ds = ds.shuffle()
except:
    pass

# %%


mode_list = list(inst_dict.keys())


# %%
model_name = "microsoft/Phi-3-medium-128k-instruct"
# model_name="OrionStarAI/Orion-14B-Chat"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
          )



# %%
while True:
    records = prepare_records(
        ds, mode_list, n_records=n_records, random_extract=False, db_name="cosmopedia")
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
