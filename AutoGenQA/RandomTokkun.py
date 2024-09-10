# %%
#コード､論理、算数などの特訓
import joblib
import random
import string
# %%
from datasets import load_dataset
#ds=load_dataset("izumi-lab/wikipedia-ja-20230720",split="train")
#title_list=ds['title']
#joblib.dump(title_list,'title_list.joblib',compress=9)
title_list=joblib.load('title_list.joblib')

# %%
import joblib

# %%
def get_random_keyword():
    return random.choice(title_list)

# %%
import joblib
import random

#keyword_list=joblib.load("oasst_noun_list.bin")

def random_algorithm0():
    classes=["JSON","YAML","CSV","タブ","表"]
    class_=random.choice(classes)
    genres=["プログラミング","コード","アルゴリズム","データ科学"]

    genre=random.choice(genres)

    keywords=get_random_keyword()
    random_ints=[random.randint(1,100) for _ in range(random.randint(1,10))]
    random_floats=[random.uniform(-1000,1000) for _ in range(random.randint(1,10))]
    random_hensu=["x","y","z"]
    length = random.randint(1, 12)
    random_hensu2 = random.sample(string.ascii_lowercase, length)

    random_numbers=random.choice([random_ints,random_floats,random_hensu,random_hensu2])
    num_strings=[str(num) for num in random_numbers]
    num_keywords=",".join(num_strings)
    
    problem=f"""あなたは{genre}の教師です｡
・アルゴリズムとコード生成に関する{class_}のタスクを生成しなさい｡
・フォーマットは厳守すること
・用いるキーワード: {keywords}
・用いる数字: {num_keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm0())



# %%
def random_algorithm1():
    classes=["大学院","大学","高校","小学校","専門学校"]
    class_=random.choice(classes)
    genres=["アルゴリズム","プログラミング","Python","html","JavaScript","C",
            "算数","数学","データベース"]
    genre=random.choice(genres)
    #keywords=random.choice(keyword_list)
    #keywords=random.sample(keywords,1)[0]
    keywords=get_random_keyword()
    problem=f"""あなたは{class_}の{genre}の教師です｡
・アルゴリズムとコード生成に関するタスクを生成しなさい｡
・フォーマットは厳守すること
・用いるキーワード: {keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm1())

# %%
def random_algorithm2():
    classes=["大学院","大学","高校","小学校","専門学校"]
    class_=random.choice(classes)
    genres=["算数","数学",]
    genre=random.choice(genres)

    keywords=get_random_keyword()
    random_ints=[random.randint(1,100) for _ in range(random.randint(1,10))]
    random_floats=[random.uniform(-1000,1000) for _ in range(random.randint(1,10))]
    random_hensu=["x","y","z"]
    length = random.randint(1, 6)
    random_hensu2 = random.sample(string.ascii_lowercase, length)

    random_numbers=random.choice([random_ints,random_floats,random_hensu,random_hensu2])
    num_strings=[str(num) for num in random_numbers]
    num_keywords=",".join(num_strings)
    
    problem=f"""あなたは{class_}の{genre}の教師です｡
・タスクを生成しなさい｡
・フォーマットは厳守すること
・用いるキーワード: {keywords}
・用いる数字: {num_keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm2())

# %%

def random_algorithm3():
    classes=["大学院","大学","高校","小学校","専門学校"]
    class_=random.choice(classes)
    genres=["論理","ロジカルシンキング","論理的思考","クイズ","謎解き",
            "コンサルタント","物語","論理学","哲学"]
    genre=random.choice(genres)
    keywords=get_random_keyword()
   
    problem=f"""あなたは{class_}の{genre}の教師です｡
・読解力と論理的思考力が要求される難解な長文の問題と模範解答を生成しなさい｡
・フォーマットは厳守すること
・用いるキーワード: {keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm3())

# %%


def random_algorithm4():
    classes=["大学院","大学","高校","小学校","専門学校"]
    class_=random.choice(classes)
    genres=["演劇","役者","ロールプレイ"]
    genre=random.choice(genres)

    keywords=get_random_keyword()
   
    problem=f"""あなたは{class_}の{genre}の教師です｡
・タスクを生成しなさい(◯◯の立場に立って、というタイプのタスクにすること)
・フォーマットは厳守すること
・シチュエーション: {keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm4())

# %%


def random_algorithm5():
    classes=["大学院","大学","高校","小学校","専門学校"]
    class_=random.choice(classes)

    keywords=get_random_keyword()

    problem=f"""あなたは{class_}の教師です｡
・高度なタスクを生成しなさい
・フォーマットは厳守すること
・シチュエーション: {keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm5())

# %%



def random_algorithm6():
    classes=["大学院","大学","高校","小学校","専門学校"]
    class_=random.choice(classes)

    keywords=get_random_keyword()

    problem=f"""あなたは{class_}の教師です｡
・厳密性を要求されるタスクを生成しなさい
・フォーマットは厳守すること
・キーワード: {keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm6())

# %%




def random_algorithm7():
    keywords=get_random_keyword()

    problem=f"""論理的な思考力を高めるための難解な問題と模範解答を生成しなさい｡
・フォーマットは厳守すること
・キーワード: {keywords}

[フォーマット]
#問題:
#回答:"""

    return problem

print(random_algorithm7())

# %%
def random_algorithm():
    return random.choice([random_algorithm0(),
        random_algorithm1(),random_algorithm2(),random_algorithm3(),random_algorithm4(),
                          random_algorithm5(),random_algorithm6(),random_algorithm7()])
random_algorithm()

# %%

# %%
# ライブラリの自動リロード
from tqdm import tqdm
import random
import pandas as pd
from src.GGUFBot import GGUFBot
from datasets import load_dataset
import json
from datetime import datetime
import joblib
import random

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"data_random_algorithm/completion_records{current_time_no_symbols}.jsonl"

print("init model")
bot = GGUFBot(model_path="../../model/Mixtral-8x22B-Instruct-v0.1.Q8_0-00001-of-00004.gguf",
              max_new_tokens=4000, n_ctx=4000, n_gpu_layers=500)
print("fin initiating model")


while True:
    record = {}
    record["database"] = "random"
    prompt=random_algorithm()
    try:
        a = bot.ask(prompt)
        if a == "":
            continue
        record["autogen_text"] = a
    except Exception as e:
        print(record, e)
        continue

    print(record)
    with open(out_path, 'a') as f:
        f.write(json.dumps(record, ensure_ascii=False)+'\n')




