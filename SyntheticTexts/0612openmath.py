from vllm import SamplingParams, LLM

from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import prepare_records
import time
import random

inst_dict = {
    "textbook": """次のデータをもとに､論理的かつ教科書調の丁寧な日本語の文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること
-[問題文]を出力
-[考え方]を出力
-[答え]を出力
-[詳細な解説]を出力

#データ
""",
    "conversation": """次のデータをもとに､論理的な日本語の会話文を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",
    "logical": """次のデータをもとに､論理的な文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること
-[問題文]を出力
-[考え方]を出力
-[答え]を出力
-[詳細な解説]を出力

#データ 
""",
    "reasoning": """次のデータをもとに､論理推定を行う文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること
-[問題文]を出力
-[考え方]を出力
-[答え]を出力
-[詳細な解説]を出力

#データ
""",
    "QandA": """次のデータをもとに､Q&Aを作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること
-[問題文]を出力
-[考え方]を出力
-[答え]を出力
-[詳細な解説]を出力

#データ
""",

}


dir_name="out_data_openmath"

wait_time = random.randint(1, 60)
time.sleep(wait_time)



# バッチサイズ
n_records = 300


os.system(f"mkdir -p {dir_name}")
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{dir_name}/model_{current_time_no_symbols}_openmath.jsonl"

ds_list = [
    # load_dataset("hpprc/jawiki-wiktionary", split="train"),
    load_dataset("kunishou/OpenMathInstruct-1-1.8m-ja", split="train"),
]
# 必要な 列だけを抽出して新しいリストに追加
ds_list_filtered = [
    ds.remove_columns(
        [col for col in ds.column_names if col not in ['question_ja', 'generated_solution_ja']])
    for ds in ds_list
]

# データセットを結合
ds = concatenate_datasets(ds_list_filtered)

ds=ds.filter(lambda x: x["question_ja"] is not None and x["generated_solution_ja"] is not None)

#ds["text"] = "問題. " + ''.join(ds["question_ja"]) + "\n 解答." + ''.join(ds["generated_solution_ja"])
def add_text_column(example):
    question_text = (example["question_ja"])
    solution_text = (example["generated_solution_ja"])
    return {"text": "問題. " + question_text + "\n 解答." + solution_text}
ds = ds.map(add_text_column)
# ds = concatenate_datasets(ds_list)

# %%
try:
    ds = ds.shuffle()
except Exception as e:
    print(e)

for record in ds:
    print(record)
    break

# %%
model_name = "microsoft/Phi-3-medium-128k-instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
)



mode_list = list(inst_dict.keys())


# %%
print(len(ds), " records")

# %%
while True:
    records = prepare_records(
        ds, mode_list, n_records=n_records, random_extract=True,db_name="kunishou/OpenMathInstruct-1-1.8m-ja",
        inst_dict=inst_dict)
    
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
