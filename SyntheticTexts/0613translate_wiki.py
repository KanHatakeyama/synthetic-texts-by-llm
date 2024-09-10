# %%
from vllm import SamplingParams, LLM

from datasets import load_dataset, concatenate_datasets
from datetime import datetime
import json
import os
from src.generator import inst_dict, prepare_records
import random

# バッチサイズ
n_records = 300

dir_name="out_data_wiki_translate"
os.system(f"mkdir -p {dir_name}")
wait_time = random.randint(1, 60)
#time.sleep(wait_time)

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{dir_name}/model_{current_time_no_symbols}.jsonl"

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


try:
    ds = ds.shuffle()
except:
    pass

model_name = "microsoft/Phi-3-medium-128k-instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=20000
          )



# %%
inst_dict = {
    "textbook": """以下のテキストを教科書に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "conversation": """以下のテキストから会話文を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "logical": """以下のテキストから論理的な文章を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "Q&A": """以下のテキストからQ&Aを生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "essay": """以下のテキストから随筆を生成しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "news_article": """以下のテキストをニュース記事に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "report": """以下のテキストをレポートに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "story": """以下のテキストを物語に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "poem": """以下のテキストを詩に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "summary": """以下のテキストを要約しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "letter": """以下のテキストを手紙に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "review": """以下のテキストをレビューに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "interview": """以下のテキストをインタビュー形式に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "recipe": """以下のテキストをレシピに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "instructions": """以下のテキストを指示に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "blog_post": """以下のテキストをブログ記事に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "speech": """以下のテキストをスピーチに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "manual": """以下のテキストをマニュアルに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "guide": """以下のテキストをガイドに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "presentation": """以下のテキストをプレゼンテーションに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "advertisement": """以下のテキストを広告に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "announcement": """以下のテキストを発表文に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "fiction": """以下のテキストをフィクションに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "nonfiction": """以下のテキストをノンフィクションに書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
    "dialogue": """以下のテキストを対話形式に書き直しなさい｡必ずすべての情報を網羅し､日本語で出力すること｡見出しは出力しない｡\n#テキスト\n""",
}


mode_list = list(inst_dict.keys())


# %%
print(len(ds), " records")

# %%
while True:

    #スタイル変換
    records = prepare_records(
        ds, mode_list, n_records=n_records, random_extract=True,inst_dict=inst_dict)
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
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

