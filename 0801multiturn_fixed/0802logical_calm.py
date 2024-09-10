# %%
# %%
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
import re
import sys
args = sys.argv
import time

"""
conda activate llmeval

export CUDA_VISIBLE_DEVICES=0
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=1
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=2
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=3
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=4
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=5
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=6
nohup python 0801fixed_multiturn.py &

export CUDA_VISIBLE_DEVICES=7
nohup python 0801fixed_multiturn.py &

"""

genre_text="""論理推論(LogicalReasoning)
論理的思考
論証
論理分析
推論
論理的推理
論理クイズ(LogicalQuiz)
論理パズル
ロジックゲーム
推理クイズ
論理問題
ロジカルパズル
推理クイズ(ReasoningQuiz)
推理パズル
推理ゲーム
謎解きクイズ
推理問題
ミステリーパズル
ロジカルシンキング(LogicalThinking)
論理的思考法
ロジックシンキング
合理的思考
分析的思考
論理推論
論理クイズ
論理思考法"""

genre_list=genre_text.split("\n")
genre_list=[i for i in genre_list if i!=""]

def get_longest_phrase_length(text):
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,。！？、\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length=9999
    return max_length

def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold

n_turns=2
batch_size=100
out_dir = "0802out_logic"
#out_dir = "0724multiturn_oum"

pid = os.getpid()
seed=int(pid)+int(datetime.now().timestamp())
print("seed: ",seed)
random.seed(seed)


os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
out_path = f"{out_dir}/model_{current_time_no_symbols}_{random.randint(0,10000)}.jsonl"


# %%

model_name = "cyberagent/calm3-22b-chat"
tensor_parallel_size=1
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=4000,
          # max_model_len=7000,
         #  gpu_memory_utilization=0.9,
         tensor_parallel_size=tensor_parallel_size,
          )

def llm_gen(llm,prompt_list,temperature=0.7,top_k=50):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=1024,
            repetition_penalty=1.2,
            top_k=top_k,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


# %%
def question_to_prompt(question,role,history=[]):
    prompt=f"""<|im_start|>system
{role}
<|im_end|>"""

    if len(history)>0:
        for q,a in history:
            prompt+=f"""<|im_start|>user
{q}<|im_end|>
<|im_start|>assistant
{a}<|im_end|>"""
    prompt+=f"""<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
    return prompt


# %%
jobs="""会社員
公務員
会社員

サラリーマン
企業従業員
オフィスワーカー
会社勤め
職員

公務員

官僚
行政職員
国家公務員
地方公務員
政府職員

自営業

フリーランス
個人事業主
自営業者
独立業者
個人企業

医師

ドクター
医者
臨床医
内科医
外科医

看護師

ナース
看護士
ケアスタッフ
医療スタッフ
看護職

エンジニア

技術者
テクニシャン
技術職
開発者
エンジニアリングスタッフ

デザイナー

クリエイター
デザインアーティスト
グラフィックデザイナー
インダストリアルデザイナー
プロダクトデザイナー

教師

先生
教員
教諭
教育者
インストラクター

販売員

セールスマン
販売スタッフ
ショップスタッフ
店員
販売担当

サービス業

サービススタッフ
ホスピタリティ従事者
サービス提供者
サービス担当
顧客サービス

農業従事者

農家
農業者
農業スタッフ
農場主
農業労働者

漁業従事者

漁師
漁業者
水産業者
漁業スタッフ
海産物従事者

建設業従事者

建設作業員
建設労働者
建築スタッフ
工事従事者
建築業者

製造業従事者

工場労働者
製造スタッフ
製造作業員
生産スタッフ
製造業者

運送業従事者

配送員
トラックドライバー
運送スタッフ
物流担当
輸送業者

金融業従事者

金融マン
バンカー
ファイナンシャルアナリスト
投資アナリスト
金融専門家

保険業従事者

保険代理店
保険営業
保険アドバイザー
保険コンサルタント
保険ブローカー

不動産業従事者

不動産エージェント
不動産仲介
不動産営業
不動産コンサルタント
不動産ブローカー

IT関連業従事者

ITプロフェッショナル
システムエンジニア
ソフトウェア開発者
ITスタッフ
ネットワークエンジニア

コンサルタント

アドバイザー
コンサルティング専門家
ビジネスコンサルタント
経営コンサルタント
コンサルティングスタッフ

作家

著者
ライター
文筆家
小説家
文学者

アーティスト

美術家
画家
芸術家
創作家
アートクリエイター

ミュージシャン

音楽家
演奏家
バンドマン
音楽アーティスト
音楽プロデューサー

俳優

アクター
パフォーマー
舞台俳優
映画俳優
演技者

タレント

芸能人
パーソナリティ
テレビタレント
ラジオパーソナリティ
メディアタレント

スポーツ選手

アスリート
プロスポーツ選手
運動選手
スポーツマン
スポーツプロフェッショナル

研究者

サイエンティスト
リサーチャー
研究職
科学者
学術研究者

学生

生徒
スチューデント
大学生
学生学徒
学び手

主婦/主夫

家事従事者
ホームメイカー
家庭管理者
家事担当
ファミリーマネージャー
自営業
医師
看護師
エンジニア
デザイナー
教師
販売員
サービス業
農業従事者
漁業従事者
建設業従事者
製造業従事者
運送業従事者
金融業従事者
保険業従事者
不動産業従事者
IT関連業従事者
コンサルタント
作家
アーティスト
ミュージシャン
俳優
タレント
スポーツ選手
研究者
学生
主婦/主夫
無職
"""
job_list=jobs.split("\n")
job_list=[i for i in job_list if i!=""]

character_text="""感情的知性が高い
責任感が強い
創造的である
同情的である
好奇心が強い
優れたコミュニケーション能力を持つ
論理的思考ができる
適応能力が高い
勤勉である
強い倫理観を持つ
協力的である
細部にまで注意を払う
問題解決能力が高い
楽観的である
情熱的である
適度な自信を持つ
誠実である
客観的である
オープンマインドである
ユーモアのセンスがある
粘り強い
文化的敏感性を持つ
忍耐強い
積極的である
革新的な考え方をする
リーダーシップの素質を持つ
巻き返す能力がある
多様性を尊重する
チームプレーヤーである
謙虚である
規律性がある
迅速な対応ができる
良心的な判断をする
気配りができる
他人への敬意を払う
楽観的な見通しを持つ
自制心がある
信頼に足る
機転が利く
分析的である
積極的に学ぶ姿勢を持つ
親切である
献身的である
思いやりがある
積極的である
仕事への熱意がある
創造性に富む
熱心である
尊敬に値する
倫理観が高い
革新的なアイディアを出す
戦略的思考ができる
思いやりのある
洞察力がある
尊大な態度がない
情熱的である
積極的な姿勢を持つ
細かい配慮ができる
信頼できる
社交的である
独立心がある
常に向上心を持つ
率先して行動する
強い意志を持つ
決断力がある
思いやりの心を持つ
積極的に問題解決にあたる
創造的なアイディアを提案する
積極的に知識を習得する
親しみやすい
積極的にコミュニケーションを取る
率先して課題に取り組む
積極的に新しいことに挑戦する
強いリーダーシップを発揮する
常に学ぶ姿勢を忘れない
積極的にチームに貢献する
細かいところにも気を配る
積極的に意見を述べる
積極的に課題解決にあたる
積極的に新しいスキルを習得する
積極的にチームワークを発揮する
積極的に問題解決に挑む
積極的に新しいアイディアを出す
積極的に新しいチャレンジをする
強い責任感を持つ
細かいところにも気を配る
積極的に新しい技術を学ぶ
積極的に新しい環境に適応する
強い意志を持って物事に取り組む
積極的に新しい経験をする
荒々しい
性格の悪い
細かい
気難しい
"""
character_list=character_text.split("\n")
character_list=[i for i in character_list if i!=""]

response_text="""
説明の妥当性について、再度、注意深く検討し直してください
説明の妥当性について、再度、検討し直してください
前提条件が正しいかどうかを再度、検討し直してください
前提条件が逆になった場合、どのような結果が生じるかを考えてみてください
論証の問題点を指摘してください
ステップ・バイ・ステップで、論証の過程を検証してください
与えられた質問の妥当性について、ステップ・バイ・ステップで検証してください
与えられた質問から、本当に結論を導けるかについて、検証してください
本当ですか?
前提を大幅に変えたときの結論について再検証してください
前提を少しだけ変えたときの結論について再検証してください
前提の問題点について指摘してください
"""

response_list=response_text.split("\n")
response_list=[i for i in response_list if i!=""]

# %%


while True:
    seed=int(pid)+int(datetime.now().timestamp())
    print("seed: ",seed)
    random.seed(seed)
    parallel_conversations=[{"qid":i,"conversations":[]} for i in range(batch_size)]
    for turn_id in range(n_turns):
        print("turn_id",turn_id)
        #はじめのターンはランダムな質問
        if turn_id==0:
            prompt_list=[]
            for qid in range(len(parallel_conversations)):
                job=random.choice(job_list)
                character=random.choice(character_list)
                role=f"あなたは{job}です。{character}性格です。"
                genre=random.choice(genre_list)
                command=f"{genre}に関する問題やクイズを一つだけしてください。質問や指示のみを出力し､それ以外は何も含めないこと"
                prompt_list.append(question_to_prompt(command,role))
            print(prompt_list[:3])  
            question_list=llm_gen(llm,prompt_list)
        else:
            #2ターン目では固定された質問を使う
            #question_list=random.sample(response_list,batch_size)
            question_list=random.choices(response_list,k=batch_size)

        #解答する
        prompt_list=[]
        for qid in range(len(parallel_conversations)):
            character=random.choice(character_list)
            role=f"あなたはアシスタントです。{character}性格です。"
            command=f"次の質問に日本語で回答しなさい｡まず初めに解くための方針を立てなさい。その後ステップバイステップで考えてください。"
            prompt_list.append(question_to_prompt(question_list[qid],role,parallel_conversations[qid]["conversations"]))
        answer_list=llm_gen(llm,prompt_list,temperature=0.01,top_k=1)

        for qid in range(len(parallel_conversations)):
            parallel_conversations[qid]["conversations"].append((question_list[qid],answer_list[qid]))

    #書き出し    
    for record in parallel_conversations:
        conversation_list=[]
        text=""
        remove_flag=False
        for q,a in record["conversations"]:

            #長すぎるものは削除(壊れた出力)
            if get_longest_phrase_length(q)>100 or get_longest_phrase_length(a)>100:
                remove_flag=True
                break
            if is_abnormal_text(q) or is_abnormal_text(a):
                remove_flag=True
                break

            text+=f"""user: {q} assistant: {a}\n"""
            conversation_list.append({"role":"user","content":q})
            conversation_list.append({"role":"assistant","content":a})
        text=text.strip()
        record["text"]=text
        record["messages"]=conversation_list
        if text=="":
            continue

        record.pop("conversations")

        with open(out_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
                                                                                    




