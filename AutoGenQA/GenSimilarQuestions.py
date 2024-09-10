# %%

#ライブラリの自動リロード
import time
from src.GGUFBot import GGUFBot
from src.HFDataset import HFDataset
from src.SimpleQuestionGenerator import SimpleQuestionGenerator
from src.AnswerGenerator import AnswerGenerator
import pandas as pd
import random
import json

# %%
from datasets import load_dataset
dataset = load_dataset("minnade/chat-daily",split="train")
dataset=dataset.filter(lambda x: x["role"] == "user")

questions=[]
for d in dataset:
    questions.append(d["body"])

# %%

n_layers=300
max_new_tokens=4000
model_path="/home/hatakeyama/python/ChatServer/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf"
bot=GGUFBot(model_path,max_new_tokens=max_new_tokens,n_ctx=max_new_tokens,n_gpu_layers=n_layers)

# %%

a_gen = AnswerGenerator(bot,n_answers=2)
out_path="data/augmented_q_and_a.jsonl"

# %%
question=random.choice(questions)
question_template = f"""以下の問題・質問・指示の類題を日本語で作成してください。
・類題はもとの問題からは必ず情報を追加・修正・削除し、内容、形式、記述方式が全く異なるようにすること
・作成した内容のみを出力すること
"""
while True:
    q=question_template+question
    try:
        new_question=bot.ask(q).replace("#類題","").replace("#問題","").strip()[:3000]
        record={"question":new_question}

        a_gen(record)
        with open(out_path, 'a') as f:
            f.write(json.dumps(record,ensure_ascii=False)+"\n")
    except:
        time.sleep(4)
    question=new_question
    #question=question[:random.randint(0,len(question))]
    if random.randint(0,3)==0:
        question=question[:int(len(question)*0.8)]
    elif random.randint(0,3)==1:
        question=question[int(len(question)*0.2):]
    if random.randint(0,3)==0:
        question=random.choice(questions)



# %%



