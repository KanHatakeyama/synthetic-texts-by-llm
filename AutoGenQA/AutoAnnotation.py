import pandas as pd
import time
import random
import json
from src.GGUFEvaluator import GGUFEvaluator
import os
import random
import pandas as pd
from tqdm import tqdm
evaluator=GGUFEvaluator()


while True:

    path="hf/cleaned_data.parquet"
    df=pd.read_parquet(path)
    df.reset_index(drop=True,inplace=True)
    df["qa"]=df["question"]+df["answer"]
    records=df.to_dict(orient="records")



    random.shuffle(records)
    qa_path="hf/qa.jsonl"

    if os.path.exists(qa_path):
        qa_to_score={}
        with open(qa_path,"r") as f:
            for line in f:
                r=json.loads(line)
                qa=r["qa"]
                score=r["score"]
                qa_to_score[qa]=float(score)
    else:
        qa_to_score={}

    count=0
    for record in tqdm(records):
        count+=1

        if count>1000:
            time.sleep(30)
            break

        q=record["question"]
        a=record["answer"]
        qa=str(q)+str(a)

        if qa in qa_to_score:
            continue
        score=evaluator(q,a)
        qa_to_score[(q+a)]=score
        print(score,q,a)
        with open(qa_path,"a") as f:
            f.write(json.dumps({"score":score,"qa":qa},ensure_ascii=False)+"\n")




