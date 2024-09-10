import pandas as pd
import json
import copy
import glob
from src.clean_records import clean_question
from huggingface_hub import HfApi, logging
import os
import time
from datasets import load_dataset
from tqdm import tqdm
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def upload():
    jsonl_path_list=glob.glob('data_orca/*.jsonl')

    all_records=[]
    for jsonl_path in jsonl_path_list:
        with open(jsonl_path) as f:
            for line in f:
                record=json.loads(line)
                all_records.append(record)

    # from huggingface

    cleaned_records=[]
    for original_record in tqdm(all_records):
        record={}
        record["question"]=clean_question(original_record["question"])
        for k in ["inst_question","inst_answer_0","text"]:
            if k in original_record:
              record[k]=original_record[k]
            else:
              record[k]=""
        if "answer_0" not in original_record:
            original_record["answer_0"]=""

        if "answer_1" not in original_record:
            original_record["answer_1"]=""

        if "ans0" in original_record:
            record["answer_0"]=original_record["ans0"]
            record["answer_1"]=original_record["ans1"]
        elif "answer_0" in original_record:
            record["answer_0"]=original_record["answer_0"]
            record["answer_1"]=original_record["answer_1"]
        else:
            print("no answer found",record)
        
        if "database" in original_record:
            record["database"]=original_record["database"]
        else:
            record["database"]="misc"

        if record["answer_0"]=="":
            continue
        #2つの回答について､それぞれ別に登録する
        r1=copy.deepcopy(record)
        r1["answer"]=r1["answer_0"]
        r1.pop("answer_0")
        r1.pop("answer_1") 

        cleaned_records.append(r1)

        r2=copy.deepcopy(record)
        r2["answer"]=str(r2["answer_1"])
        r2.pop("answer_0")
        r2.pop("answer_1") 
        if len(r2["answer"])>2:
            cleaned_records.append(r2)


        #うまく質問､回答をparseできなかったケースの対応
        if record["question"]=="" and record["text"]!="":
            q_terms=["質問文:","Q:","Q：","**Q:**","**質問:**","質問",]
            a_terms=["回答文","A:","A：","**A**","答案","**回答:**","答え" ,"回答",]
            for q_term in q_terms:
                if q_term in record["text"]:
                    qa_pairs = record["text"].split(q_term)
                    break

            q_a_list=[]
            for pair in qa_pairs:
                for a_term in a_terms:
                    if a_term in pair:
                        try:
                            qa = pair.split(a_term)
                            q, a = qa[0],qa[1:]
                            a=a_term.join(a)
                        except Exception as e:
                            print(qa,e)
                            continue
                        q=q.strip()
                        a=a.strip()
                        for k in [":","：","1:","2:","3:"]:
                            if q.startswith(k):
                                q=q[len(k):]
                            if a.startswith(k):
                                a=a[len(k):]
                        q_a_list.append((q.strip(),a.strip()))
                        break
            for qa in q_a_list:
                r=copy.deepcopy(record)
                r["question"]=qa[0]
                r["answer"]=qa[1]
                cleaned_records.append(r)
    df=pd.DataFrame(cleaned_records)
    #シャッフル
    #df=df.sample(frac=1).reset_index(drop=True)
    parquet_path="hf/cleaned_data.parquet"
    df.to_parquet(parquet_path)
    df.to_csv("hf/cleaned_data.csv")



    logging.set_verbosity_debug()
    hf = HfApi()
    hf.upload_file(path_or_fileobj=parquet_path,
                    path_in_repo=f"1.parquet",
                    repo_id="kanhatakeyama/OrcaJaMixtral8x22b", repo_type="dataset")





if __name__ == "__main__":
    while True:
        try:
            upload()
            print("uploaded")
            break
            time.sleep(3600*3)
        except Exception as e:
            print("error",e)
            time.sleep(600)


