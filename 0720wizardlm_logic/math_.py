# %%
# %%
# %%
from llama_cpp import Llama
from datetime import datetime
import json
import os
import random
import time

for i in range(109):
    random.seed(datetime.now().time().microsecond+random.randint(0,10000))



# %%

wait_time = random.randint(1, 10)
time.sleep(wait_time)

#####################
# 設定関連
n_records = 1
out_dir = "0722math"

#tsubame
model_path="../model/WizardLM-2-8x22B.Q8_0-00001-of-00005.gguf"

################
# メイン

os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}.jsonl"

def is_abnormal_eng_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    period_count = text.count('.')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

class GGUFBot:
    def __init__(self, model_path="model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf",
                 max_new_tokens=4000,
                 n_gpu_layers=100,
                 n_ctx=4096) -> None:
        print("loading model...")

        self.model = Llama(model_path=model_path,
                           n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, )
        self.max_new_tokens = max_new_tokens

    def ask(self, question,temperature=0.01):

        prompt = f"""<s>[INST]{question}[/INST]"""

        output = self.model(
            prompt,
            max_tokens=self.max_new_tokens,
            temperature = temperature,
            # top_p = 0.8,
            # repeat_penalty = 1.1,
            # frequency_penalty = 1.0,
            # presence_penalty = 1.0,
            # stop = ["\n###  Instruction:", "\n### Response:", "\n"],
            # echo = True,
        )
        return output["choices"][0]["text"].strip()


# %%


bot = GGUFBot(model_path=model_path)


# %%


print("initiated llm")

# %%


genre_texts="""
幾何学
    1年生
        数と計算
            1から10までの数
            数の数え方
            簡単な足し算と引き算
        形と空間
            基本的な形の認識（円、三角形、四角形など）
            立体の形の認識（球、立方体など）
        時間と測定
            時間の読み方（時計）
            長さ、重さ、容量の概念
        図形の作成
            点と線の描き方
            簡単な図形の描画

    2年生
        数と計算
            20までの数
            簡単な掛け算と割り算の導入
        形と空間
            図形の種類と特徴
            図形の合成と分解
        時間と測定
            より具体的な時間の計測
            長さ、重さ、容量の具体的な測定
        パターンと規則性
            数列の規則性
            簡単なパターンの認識

    3年生
        数と計算
            100までの数
            基本的な掛け算と割り算
        形と空間
            平面図形の特徴
            立体図形の特徴
        時間と測定
            時間の計算
            面積と体積の概念
        パターンと規則性
            より複雑なパターンの認識

    4年生
        数と計算
            1000までの数
            割り算の筆算
        形と空間
            平行と垂直の概念
            図形の合同と対称
        時間と測定
            時間の計測と表示
            面積と体積の測定
        データの扱い
            表とグラフの読み方
            簡単な統計の概念

    5年生
        数と計算
            分数と小数の計算
            割り算の複雑な計算
        形と空間
            平面図形の性質
            立体図形の展開図
        時間と測定
            複雑な時間の計算
            より具体的な面積と体積の測定
        データの扱い
            平均と中央値の概念
            グラフの作成と解釈

    6年生
        数と計算
            分数、小数、百分率の計算
            比と比例の概念
        形と空間
            平面図形と立体図形の詳細
            図形の面積と体積の計算
        時間と測定
            速度と距離の計算
            複雑な面積と体積の測定
        データの扱い
            統計的なデータの収集と解析
            より複雑なグラフの作成と解釈

中学校の数学単元

    1年生
        数と計算
            正負の数
            四則演算の応用
        方程式
            一次方程式
        図形
            基本的な幾何
            角度と三角形の性質
        関数
            比例と反比例の概念
        データの扱い
            基本的な統計の応用
            表とグラフの詳細

    2年生
        数と計算
            有理数と無理数
            べき乗と根
        方程式
            連立方程式
        図形
            平行線と面積
            円の性質
        関数
            一次関数の応用
        データの扱い
            統計のさらなる応用
            確率の基礎

    3年生
        数と計算
            実数の概念
            多項式の計算
        方程式
            二次方程式
        図形
            円錐、球、円柱の性質
            相似と合同の応用
        関数
            二次関数
        データの扱い
            確率の応用
            複雑な統計の解析
"""

genre_list=genre_texts.split("\n")
genre_list=[i.strip() for i in genre_list if i!=""]

def generate_random_natural_numbers():
    num_count = random.randint(1, 5)
    random_numbers = [random.randint(1, 100) for _ in range(num_count)]
    return random_numbers


# %%


def prepare_seed_prompt():

    level = random.choice(["中学校", "高校",
                            ])
    genre=random.choice(genre_list)
    prompt=f"""{level}レベルの数学の問題を1つだけ生成せよ｡問題文は日本語で出力すること｡
- 単元: {genre}
- 出力には問題のみを含め､それ以外は一切を含めないこと"""

    if random.random()>0.5:
        random_numbers=generate_random_natural_numbers()
        prompt+=f"""\n- 次の数値を使って問題を作成すること{random_numbers}\n"""
    return prompt


while True:

    prompt=prepare_seed_prompt()
    q=bot.ask(prompt,temperature=1.0)

    a=bot.ask(f"次の問題に解答しなさい｡{q}",temperature=0.01)
    a2=bot.ask(f"以下のやり取りが正しいか､ステップバイステップで慎重に検証しなさい｡ Q.{q}\n A.{a}\n#検証結果",temperature=0.01)

    record={
        "instruction":q,
        "output":a,
        "output2":a2,
        "text":f"{q}\n{a}\n{a2}"
    }

    # print("saving to "+out_path)
    with open(out_path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# %%



