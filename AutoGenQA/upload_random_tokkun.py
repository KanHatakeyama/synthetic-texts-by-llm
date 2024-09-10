# %%
import glob
import json

import pandas as pd
jsonl_dirs=glob.glob('data_random_algorithm/*.jsonl')
len(jsonl_dirs)

# %%
records=[]
for jsonl_dir in jsonl_dirs:
    with open(jsonl_dir) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception as e:
                print(e)
                print(line)
len(records)

# %%
cleaned_records=[]
record=records[0]

txt=record['autogen_text']
def parse_qa(txt):
    q_template="#問題:"
    a_template="#回答:"

    a_pos=txt.find(a_template)
    q_pos=txt.find(q_template)

    if a_pos==-1 or q_pos==-1:
        return None,None
    if txt.find("#問題1")>0:
        return None,None
    a=txt[a_pos+len(a_template):].strip()
    q=txt[q_pos+len(q_template):a_pos].strip()
    if a=="" or q=="":
        return None,None
    return q,a

# %%
cleaned_records=[]
for record in records:
    txt=record['autogen_text']
    q,a=parse_qa(txt)
    if q is not None and a is not None:
        cleaned_records.append({'question':q,'answer':a})

# %%
df=pd.DataFrame(cleaned_records)
parquet_path='data_random_algorithm/1.parquet'

# %%
df.to_parquet(parquet_path)

# %%

from huggingface_hub import HfApi, logging
hf = HfApi()
hf.upload_file(path_or_fileobj=parquet_path,
                path_in_repo=f"1_tsubame.parquet",
                repo_id="kanhatakeyama/LogicalDatasetsByMixtral8x22b", repo_type="dataset")



# %%



