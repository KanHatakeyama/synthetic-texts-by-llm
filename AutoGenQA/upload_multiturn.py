# %%
# マルチターンデータセットを統合する

# %%
from huggingface_hub import HfApi, logging
import pandas as pd
import datasets
import glob
import json
import re
jsonl_path_list = glob.glob("data_multi_paraph*/*.jsonl")

# %%

records = []
for jsonl_path in jsonl_path_list:
    with open(jsonl_path, "r") as f:
        for line in f:
            records.append(json.loads(line))

# %%
jsonl_path

# %%

# きれいにする
remove_words = [
    "User:",
    "Assistant:",
    "ユーザー：",
    "アシスタント：",
    "ユーザー:",
    "アシスタント:",
]


def clean_chat(txt):
    for word in remove_words:
        if txt.startswith(word):
            txt = txt[len(word):]
    txt = txt.strip()
    if txt[0] == "「" and txt[-1] == "」":
        txt = txt[1:-1]

    return txt


# %%
txt = "「アートの世界において、巨匠と呼ばれる人物は誰がありますか？」"
clean_chat(txt)

# %%


def parse_full_auto_dialogue(txt):
    if "### 指示1:" in txt and "### 応答1:" and "### 指示2:" in txt and "### 応答2:":
        # 正規表現で「指示」と「応答」を抽出
        pattern = r"### 指示(\d+):(.*?)\n### 応答\1:(.*?)(?=\n### 指示|$)"
        matches = re.findall(pattern, txt, re.DOTALL)

        # 辞書に変換
        dialogue_dict = {}
        add_flag = True
        for match in matches:
            idx = int(match[0])
            if idx >= 3:
                continue
            user_text = match[1].strip()
            assistant_text = match[2].strip()
            if len(user_text) < 3:
                add_flag = False
                break
            if assistant_text == "":
                add_flag = False
                break

            if user_text.find("以下の情報を元に、UserとAssistantのやりとりを") >= 0:
                add_flag = False
                break

            dialogue_dict[f"q{idx}"] = clean_chat(user_text)
            dialogue_dict[f"a{idx}"] = clean_chat(assistant_text)

        # 最後にチェック
        if "q1" in dialogue_dict and "q2" in dialogue_dict and "a1" in dialogue_dict and "a2" in dialogue_dict:
            return dialogue_dict, add_flag

    return {}, False


# %%
record = records[1]
dialogues = []

invalid_records = []
for record in records:

    # 自動生成のdialogue
    if "autogen_text" in record:
        txt = record["autogen_text"].strip()

    # 質問を与える場合
    elif "question" in record and "response" in record:
        txt = record["response"].strip()
    else:
        invalid_records.append(record)
        print("invalid record:",record)
        # raise ValueError(record)

    dialogue_dict, add_flag = parse_full_auto_dialogue(txt)
    dialogue_dict["database"] = record["database"]
    if add_flag:
        dialogues.append(dialogue_dict)

# %%

# %%
record

# %%
df = pd.DataFrame(dialogues)
df = df.reindex()


jsonl_path = "data/multi.jsonl"
with open(jsonl_path, "w") as f:
    for dialogue in dialogues:
        f.write(json.dumps(dialogue, ensure_ascii=False)+"\n")

hf = HfApi()
logging.set_verbosity_debug()
hf.upload_file(  # path_or_fileobj=parquet_path,
    path_or_fileobj=jsonl_path,
    # path_in_repo=f"1.parquet",
    path_in_repo=f"1_t.jsonl",
    repo_id="kanhatakeyama/AutoMultiTurnByMixtral8x22b", repo_type="dataset")

# %%

df.reindex()

# %%
