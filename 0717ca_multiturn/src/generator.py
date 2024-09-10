import random


def extract_random_part(text):
    text_length = len(text)
    extract_length = min(text_length, random.randint(400, 2000))
    start_index = random.randint(0, text_length - extract_length)
    return text[start_index:start_index + extract_length]


inst_dict = {
    "textbook": """次のデータをもとに､論理的かつ教科書調の丁寧な日本語の文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",
    "conversation": """次のデータをもとに､論理的な日本語の会話文を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",
    "logical": """次のデータをもとに､論理的な文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ 
""",
    "reasoning": """次のデータをもとに､論理推定を行う文章を作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",
    "QandA": """次のデータをもとに､Q&Aを作成しなさい｡
-事実を正確に守り､推測出来ない事項については記述しないこと｡
-元の文章の流用は避け､表現や段落分け､文体などを必ず変更すること｡
-必ず日本語で出力すること

#データ
""",

}


def prepare_records(ds, mode_list,
                    random_extract=True,
                    n_records=300,
                    db_name="",
                    inst_dict=inst_dict,
                    ):
    ds = ds.shuffle()

    records = []
    cnt = 0
    for record in ds:
        #print(record)
        mode = random.choice(mode_list)
        inst = inst_dict[mode]

        # cosmopedia
        if "prompt" in record:
            key = random.choice(["prompt", "text"])
        else:
            key = "text"

        text = record[key]
        if random_extract:
            text = extract_random_part(text)
        text = f"""<|user|>
{inst}{text}<|end|>
<|assistant|>"""

        if "url" not in record:
            assert db_name != "", "url not found. you should set db_name"
            record["url"] = db_name
        records.append(
            {"original_text": text,
                "mode": mode,
                "url": record["url"]
             }
        )
        cnt += 1
        if cnt > n_records:
            break

    return records
