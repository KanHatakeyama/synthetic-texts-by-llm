# %%
# %%
# %%
# %%
from categories import *
import time
import os
from datetime import datetime
from vllm import SamplingParams, LLM
import json
import random
import re
import sys
args = sys.argv
"""
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7
conda activate llmeval
python gen.py

"""


def get_longest_phrase_length(text):
    # 区切り文字として、スペース、カンマ、句読点、改行を指定
    delimiters = r'[ ,。！？、\n]'
    # テキストを区切り文字で分割
    try:
        phrases = re.split(delimiters, text)
        # 最大のフレーズの長さを取得
        max_length = max(len(phrase) for phrase in phrases)
    except:
        max_length = 9999
    return max_length


def is_abnormal_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    # 複数の区切り文字をカウント
    period_count = text.count('.') + text.count(',') + \
        text.count('､') + text.count('｡')
    ratio = word_count / period_count if period_count > 0 else word_count
    return ratio > threshold


def is_good_sentence(sentence):
    if get_longest_phrase_length(sentence) > 100:
        return False
    if is_abnormal_text(sentence):
        return False
    return True


batch_size = 10
max_count = 10**5
out_dir = "0805wizard8x22b"

pid = os.getpid()
seed = int(pid)+int(datetime.now().timestamp())
print("seed: ", seed)
random.seed(seed)

# %%


# %%


# %%

# %%
os.system(f"mkdir -p {out_dir}")

current_time_no_symbols = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S").replace("-", "").replace(":", "").replace(" ", "")
rand_id = random.randint(0, 10000)


# %%

model_name = "alpindale/WizardLM-2-8x22B"
tensor_parallel_size = 4
llm = LLM(model=model_name, trust_remote_code=True,
          max_model_len=4000,
          # max_model_len=7000,
          #  gpu_memory_utilization=0.9,
          tensor_parallel_size=tensor_parallel_size,
          )


def llm_gen(llm, prompt_list, temperature=0.7, top_k=50):

    outputs = llm.generate(
        prompt_list,
        sampling_params=SamplingParams(
            temperature=temperature,
            max_tokens=2048,
            repetition_penalty=1.1,
            top_k=top_k,
        )
    )
    return [i.outputs[0].text.strip() for i in outputs]


# %%
def question_to_prompt(question, role="an artificial intelligence assistant"):
    prompt = f"""A chat between a curious user and {role}. The assistant gives helpful, 
detailed, and polite answers to the user's questions. USER: {question} ASSISTANT:"""
    return prompt


def question_to_prompt_2nd_turn(q1, a1, q2, role="an artificial intelligence assistant"):
    prompt = f"""A chat between a curious user and {role}. The assistant gives helpful, 
detailed, and polite answers to the user's questions. USER: {q1} ASSISTANT: {a1} USER: {q2} ASSISTANT:"""
    return prompt


# %%
additional_question_commands = [
    "For the following Q&A, generate an additional short question that change the assumptions.",
    "For the following Q&A, generate an additional short question.",
    "For the following Q&A, generate an additional short question that is related to the previous question.",
    "For the following Q&A, generate an additional short question that is related to the previous question but is more general.",
    "For the following Q&A, generate an additional short question that is related to the previous question but is more specific.",
    "For the following Q&A, generate an additional short question that is related to the previous question that doubts the assumptions.",
    "For the following Q&A, generate an additional short question that is related to the previous question that doubts the conclusions.",
    "For the following Q&A, generate an additional short question that is related to the previous question that is more complex.",
    "For the following Q&A, generate an additional short question that introduces a new perspective.",
    "For the following Q&A, generate an additional short question that challenges the premises.",
    "For the following Q&A, generate an additional short question that supports the answer with further details.",
    "For the following Q&A, generate an additional short question that requires a different type of reasoning.",
    "For the following Q&A, generate an additional short question that explores potential exceptions.",
    "For the following Q&A, generate an additional short question that examines the implications.",
    "For the following Q&A, generate an additional short question that suggests an alternative solution.",
    "For the following Q&A, generate an additional short question that tests the robustness of the answer.",
    "For the following Q&A, generate an additional short question that considers a counterexample.",
    "For the following Q&A, generate an additional short question that connects the topic to a broader context.",
    "For the following Q&A, generate an additional short question that requires applying the concept to a practical scenario.",
    "For the following Q&A, generate an additional short question that probes deeper into the underlying principles.",
    "For the following Q&A, generate an additional short question that explores a related subtopic.",
    "For the following Q&A, generate an additional short question that contrasts with the initial question.",
]

# %%

count = 0
file_id = 0
while True:
    seed = int(pid)+int(datetime.now().timestamp())
    print("seed: ", seed)
    random.seed(seed)
    prompt_list = []

    # q1
    for qid in range(batch_size):
        job = random.choice(job_list)
        character = random.choice(character_list)
        role = f"{job}. You are {character}"
        genre = random.choice(genre_list)+","+random.choice(genre_list)
        level = random.choice(levels)
        quiz_type = random.choice(
            ["mathematical problem", "reasoning quiz",
             "logical quiz", "coding problem",
             "logical puzzle", "reasoning puzzle",

             ])
        command = f"""Prepare a {quiz_type}.
- Output only the question, which is not too long.
- NEVER the answer, hints, or any other things.
- Topic: {genre}.
- Level: {level}.
"""
        prompt_list.append(question_to_prompt(command, role))

    print(prompt_list[:3])
    first_question_list = llm_gen(llm, prompt_list, temperature=0.01)

    # a1
    prompt_list = [question_to_prompt(q) for q in first_question_list]
    first_answer_list = llm_gen(llm, prompt_list, temperature=0.01)

    # q2
    prompt_list = []
    for q1, a1 in zip(first_question_list, first_answer_list):
        command = random.choice(additional_question_commands)
        question = f"""{command}
- Never output answer, only output question.
[Q&A start]
Question: {q1}
Answer: {a1}
[Q&A end]"""
        prompt_list.append(question_to_prompt(question))

    second_question_list = llm_gen(llm, prompt_list, temperature=0.7)

    # a2
    prompt_list = [question_to_prompt_2nd_turn(q1, a1, q2) for q1, a1, q2 in zip(
        first_question_list, first_answer_list, second_question_list)]
    second_answer_list = llm_gen(llm, prompt_list, temperature=0.01)

    out_path = f"{out_dir}/model_{current_time_no_symbols}_{rand_id}_{file_id}.jsonl"

    # 書き出し

    with open(out_path, "a") as f:
        for q1, a1, q2, a2 in zip(first_question_list, first_answer_list, second_question_list, second_answer_list):
            covnersation_list = []

            # 1st turn
            if not is_good_sentence(q1):
                continue
            if not is_good_sentence(a1):
                continue

            covnersation_list.append({"role": "user", "content": q1})
            covnersation_list.append({"role": "assistant", "content": a1})

            # 2nd turn
            if is_good_sentence(q2) and is_good_sentence(a2) and len(q2) > 10 and len(a2) > 10:
                covnersation_list.append({"role": "user", "content": q2})
                covnersation_list.append({"role": "assistant", "content": a2})

            record = {}
            record["messages"] = covnersation_list

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# %%
