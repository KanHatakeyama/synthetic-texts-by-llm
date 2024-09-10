# %%
#ライブラリの自動リロード
from tqdm import tqdm
import random
import pandas as pd
from src.GGUFBot import GGUFBot
from src.AnswerGenerator import AnswerGenerator
from datasets import load_dataset
import json
import os
import copy
from datetime import datetime
import time

time.sleep(random.randint(0, 15))
current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
#out_path = f"data_multi_misc/completion_records{current_time_no_symbols}.jsonl"

out_path=f"data_orca/completion_records_{current_time_no_symbols}.jsonl"
# %%

print("init original dataset")
ds=load_dataset("atsushi3110/cross-lingual-openorcha-830k-en-ja",split="train")
df=pd.DataFrame(ds)
df["database"]="atsushi3110/cross-lingual-openorcha-830k-en-ja_"+df["id/en"]
df["question"]=df["question/ja"]
df=df.drop(columns=["response/en","system_prompt/en","question/ja","response/ja","question/en","id/en"],axis=1)
records=df.to_dict(orient='records')
 
def load_questions():
    if os.path.exists(out_path):
        print("loading done questions")
        done_df=pd.read_json(out_path, orient='records', lines=True)
        done_questions=list(done_df["question"].values)
    else:
        done_questions=[]

    return done_questions
    #undone_questions=[]
    #for record in tqdm(records):
    #    if record["question"] not in done_questions:
    #        undone_questions.append(record)
    #return undone_questions,done_questions

#print("loading undone questions")
#ecords,done_questions=load_questions()

#bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf", 
#              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=400)
#bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q6-00001-of-00004.gguf", 
#              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=400)
print("init model")
bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf", 
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=400)
print("fin initiating model")


a_gen = AnswerGenerator(bot,n_answers=1)
while True:
    #count=0
    random.shuffle(records)
    #print("start loop. loading done questions")
    #done_questions=load_questions()
    #records,done_questions=load_questions()
    #print("loaded done questions")
    #print("start loop")

    #if len(records)==0:
    #    print("everything finished")
    #    break
    for record in tqdm(records):
        print(record)
        if "q" in  record.keys():
            if type(record["q"]) is not str:
                continue
            record["question"]=record["q"]
            record["answer"]=record["a"]
            #print(record)
        #if record["question"] in done_questions:
        #    print(f"skip {record['question']}")
        #    #records.remove(record)
        #    continue
        try:
            a_gen(record)
        except Exception as e:
            print(record,e)
        #record["answer_1"]=copy.copy(record["answer"])
        #record.pop("answer")
        with open(out_path, 'a') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')

        #count+=1
        #if count>1000:
        #    break
