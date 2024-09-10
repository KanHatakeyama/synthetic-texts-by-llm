# %%


"""
export CUDA_VISIBLE_DEVICES=0
nohup python translate.py > log0.txt &
"""

from datasets import load_dataset
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
from datasets import load_dataset
import glob
import time
import re

# %%
#pidを取得
import os
pid=os.getpid()
random.seed(datetime.now().time().microsecond+int(pid))

def get_longest_phrase_length(text):
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,。！？、\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length=9999
    return max_length

def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold


# %%
in_dir="0725wizard_7b"
out_dir = "0802out_trans"
####################

batch_size=100
################
# メイン

os.system(f"mkdir -p {out_dir}")
rand_num=random.randint(0,1000000)
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}_{rand_num}.jsonl"


# %%
model_name = "cyberagent/calm3-22b-chat"

#model_name="nitky/Oumuamua-7b-instruct-v2"

#model_name = "hatakeyama-llm-team/Tanuki-8B-Instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=6000,
          # max_model_len=7000,
           #gpu_memory_utilization=0.4,
          )

# %%

def question_to_prompt(question):
    return f"""<|im_start|>system
あなたは翻訳家です。<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""


def llm_gen(llm,prompt_list):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=0.7,
            max_tokens=2048,
            repetition_penalty=1.05,
            top_k=50,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]

# %%
persona_list=[
    "次のテキストを忠実に日本語に訳しなさい｡",
    "次のテキストを忠実にカジュアルな日本語に訳しなさい｡",
    "次のテキストを忠実にフォーマルな日本語に訳しなさい｡",
    "次のテキストを忠実に教科書調の日本語に訳しなさい｡",
    "次のテキストを忠実に教科書調の日本語に訳しなさい｡訳した後、内容の正当性を吟味するディスカッションをしなさい",
    "次のテキストを忠実に教科書調の日本語に訳しなさい｡訳した後、追加でQ&Aを生成しなさい。",
    "次のテキストをもとに、日本語で会話文を生成しなさい",
    "次のテキストをもとに、日本語でQ&Aを生成しなさい",
]


# %%

while True:
    random.seed(datetime.now().time().microsecond+int(pid))
    try:
        input_files=glob.glob(f"{in_dir}/*.jsonl")
        inpu_file_path=random.choice(input_files)
        ds=load_dataset("json", data_files=inpu_file_path,split="train")
    except Exception as e:
        print(e)
        time.sleep(10)
        continue

    n_records=len(ds)
    print(n_records,"records")

    q_list=[]
    id_list=[]
    eng_list=[]
    for i in range(batch_size):
        record_id=random.randint(0,n_records)
        try:
            record=ds[record_id]
        except Exception as e:
            print(e)
            continue

        text=record["text"]
        eng_list.append(text)
        persona=random.choice(persona_list)
        text=f"{persona}必ず全ての情報を網羅し､翻訳文以外は何も出力しないこと｡\n[英文開始]\n{text}[英文終了]\n#日本語の翻訳\n"
        q_list.append(text)
        id_list.append(record_id)
        
    # %%


    prompt_list=[question_to_prompt(text) for text in q_list]
    answer_list=llm_gen(llm,prompt_list)

    # %%
    # %%
    for i in range(len(q_list)):
        record={
            "text":answer_list[i]+"\n"+eng_list[i],
            "id":id_list[i],
        }

        reject_flag=False
        for key in record:
            if key!="text":
                continue
            q=record[key]
            if get_longest_phrase_length(q)>100:
                reject_flag=True
                break

            #record[key]=clean(record[key])
            if record[key]=="":
                reject_flag=True
                break
        if reject_flag:
            continue

        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

