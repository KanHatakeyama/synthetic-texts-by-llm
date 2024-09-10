# %%
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
from src.clean_utils import clean
import random
from genres import prepare_prompt

# %%
out_dir = "0719out"

for i in range(109):
    random.seed(datetime.now().time().microsecond+random.randint(0,10000))

batch_size=1000
#batch_size=10

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}_{random.randint(0,10000)}.jsonl"



# %%
model_name = "cyberagent/calm3-22b-chat"

#model_name = "hatakeyama-llm-team/Tanuki-8B-Instruct"
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=4000,
          # max_model_len=7000,
         #  gpu_memory_utilization=0.9,
          )

def llm_gen(llm,prompt_list):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=0.1,
            max_tokens=1024,
            repetition_penalty=1.2,
            top_k=50,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


# %%


def question_to_prompt(question):
    return f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""



# %%
#応答文の指示

response_inst_text="""
以下のやりとりをもとに､追加の質問を1つ生成しなさい
以下のやりとりをもとに､内容を深く掘り下げる､追加の質問を1つ生成しなさい
以下のやりとりをもとに､回答に反対する旨の､追加の質問を1つ生成しなさい
以下のやりとりをもとに､回答に賛成する旨の､追加の質問を1つ生成しなさい
以下のやりとりをもとに､回答に疑問を呈する旨の､追加の質問を1つ生成しなさい
以下のやりとりをもとに､話題を変えたい旨の､追加の質問を1つ生成しなさい
以下のやりとりをもとに､回答の信頼性を再確認する旨の追加の質問を1つ生成しなさい
以下のやりとりをもとに、さらなる関連情報を求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、異なる視点からの追加の質問を1つ生成しなさい
以下のやりとりをもとに、実践的な応用についての追加の質問を1つ生成しなさい
以下のやりとりをもとに、具体的な例を求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、データや統計を求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、関連する最新の研究やニュースを求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、回答の前提を確認する追加の質問を1つ生成しなさい
以下のやりとりをもとに、回答の影響を考慮する追加の質問を1つ生成しなさい
以下のやりとりをもとに、回答の長所を強調する追加の質問を1つ生成しなさい
以下のやりとりをもとに、回答の短所を指摘する追加の質問を1つ生成しなさい
以下のやりとりをもとに、将来的な展望についての追加の質問を1つ生成しなさい
以下のやりとりをもとに、異なる分野への応用可能性についての追加の質問を1つ生成しなさい
以下のやりとりをもとに、倫理的な観点からの追加の質問を1つ生成しなさい
以下のやりとりをもとに、具体的な手順や方法を詳細に求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、他の研究や意見と比較する追加の質問を1つ生成しなさい
以下のやりとりをもとに、理論的な基盤を確認する追加の質問を1つ生成しなさい
以下のやりとりをもとに、具体的な成功事例を求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、失敗事例やリスクについての追加の質問を1つ生成しなさい
以下のやりとりをもとに、関連する実験結果を求める追加の質問を1つ生成しなさい
以下のやりとりをもとに、社会的な影響についての追加の質問を1つ生成しなさい
以下のやりとりをもとに、環境への影響を考慮する追加の質問を1つ生成しなさい
以下のやりとりをもとに、費用対効果についての追加の質問を1つ生成しなさい
以下のやりとりをもとに、トレンドや未来予測についての追加の質問を1つ生成しなさい
"""
response_inst_list=response_inst_text.strip().split("\n")
response_inst_text=[i.strip() for i in response_inst_list if i.strip()!=""]

# %%
ans_style_text="""
あなたは誠実なアシスタントです｡次の質問に回答しなさい｡
あなたは誠実なアシスタントです｡次の質問に端的に回答しなさい｡
あなたは誠実なアシスタントです｡次の質問に丁寧に回答しなさい｡
あなたは誠実なアシスタントです｡次の質問にステップバイステップで回答しなさい｡
"""
ans_style_list=ans_style_text.strip().split("\n")
ans_style_list=[i.strip() for i in ans_style_list if i.strip()!=""]


def filter_record_list(record_list,final_check=False):
    filtered=[]
    for record in record_list:
        reject_flag=False
        for key in record:
            if key=="id":
                continue
            if record[key]=="":
                reject_flag=True
                break
            if final_check:
                if record[key] is None:
                    reject_flag=True
                elif len(record[key])<10:
                    reject_flag=True
        if reject_flag:
            continue
        filtered.append(record)
    return filtered



# %%
while True:

    # %%

    record_list=[]

    for i in range(batch_size):
        d={
        "id":i,
        "q1":".",
        "a1":".",
        "q2":".",
        "a2":".",
        }
        record_list.append(d)

    prompt_list=[]

    #種instのランダムキーワード
    for i in range(batch_size):
        inst,_=prepare_prompt(random.randint(0,100000))
        prompt= question_to_prompt(f"{inst}")
        prompt_list.append(prompt)

    #質問1の生成
    first_instruction_list=llm_gen(llm,prompt_list)
    for i,question in enumerate(first_instruction_list):
        q=clean(question)
        record_list[i]["q1"]=q


    record_list=filter_record_list(record_list)
    first_instruction_list=[record["q1"] for record in record_list]
    prompt_list=[random.choice(ans_style_list)+f"{question}" for question in first_instruction_list]

    #回答1の生成
    first_answer_list=llm_gen(llm,prompt_list)
    for i,answer in enumerate(first_answer_list):
        answer=clean(answer)
        record_list[i]["a1"]=answer

    record_list=filter_record_list(record_list)


    #質問2の生成
    prompt_list=[]
    for record in record_list:
        inst=random.choice(response_inst_list)
        q1=record["q1"]
        a1=record["a1"]
        question=f"{inst}｡質問のみを出力し､それ以外はなにも出力しないこと｡\nQ.{q1}\nA.{a1}"
        prompt_list.append(question_to_prompt(question))

    second_instruction_list=llm_gen(llm,prompt_list)

    for i,inst in enumerate(second_instruction_list):
        record_list[i]["q2"]=clean(inst)
    record_list=filter_record_list(record_list)

    #回答2の生成
    prompt_list=[question_to_prompt(record["q2"]) for record in record_list]
    second_answer_list=llm_gen(llm,prompt_list)

    for i,inst in enumerate(second_answer_list):
        record_list[i]["a2"]=clean(inst)
    record_list=filter_record_list(record_list,final_check=True)

    for record in record_list:
        record.pop("id")
        record["text"]=f"user: {record['q1']}\nassistant: {record['a1']}\nuser: {record['q2']}\nassistant: {record['a2']}"

    # %%
    for record in record_list:
        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
