import argparse
import json
from src.GGUFBot import GGUFBot
from src.HFDataset import HFDataset
import random, string

def randomname(n=5):
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)

inst_list=[
"回答文は丁寧であること"
"回答文は丁寧であること"
"回答文は丁寧であること"
"回答文は簡潔であること"
"回答文はstep by stepで作文してください",
"回答文はstep by stepで作文してください",
"回答文はstep by stepで作文してください",
]

def gen_prompt(inst,text):
    prompt_template=f"""次の文章をもとに､日本語の質問文と日本語の回答文をそれぞれ一つ生成しなさい
    #制約
    {inst}
    #文章
    {text}
    #質問文と回答文
    """
    return prompt_template

def gen_qa(r):


    q,a=r.split("回答文")
    q=q.replace("質問文","").strip()
    a=a.strip()
    if a[0]=="：":
        a=a[1:]
    if q[0]=="：":
        q=q[1:]
    if a[0]==":":
        a=a[1:]
    if q[0]==":":
        q=q[1:]

    a=a.strip()
    q=q.strip()
    return q,a


def main(args):
    ds = HFDataset(args.ds_name, streaming=True)
    for _ in range(args.preload_iter):
        next(ds)

    bot = GGUFBot(args.model_path, max_new_tokens=args.max_new_tokens, n_ctx=args.max_new_tokens, n_gpu_layers=args.n_layers)

    rand_name=randomname()
    save_path = f"data/directQA_{args.ds_name.replace('/', '_')}_{rand_name}.jsonl"
    while True:
        try:
            record = {}
            record["text"] = next(ds)
            record["database"] = args.ds_name
            prompt=gen_prompt(random.choice(inst_list),record["text"][:1000])
            r=bot.ask(prompt)

            record["text"]=""
            try:
                q,a=gen_qa(r)
            except Exception as e:
                print("error: invalid Q&A fromat", e)
                record["text"]=r
                q,a="",""
            record["question"]=q
            record["answer_0"]=a

            with open(save_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print("error", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with AI model.")
    parser.add_argument("--ds_name", type=str, default="hatakeyama-llm-team/WikiBookJa", help="Name of the dataset")
    parser.add_argument("--model_path", type=str, default="/home/hatakeyama/python/ChatServer/model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf", help="Path to the model")
    parser.add_argument("--n_layers", type=int, default=400, help="Number of model layers to be loaded on GPU")
    parser.add_argument("--max_new_tokens", type=int, default=4000, help="Maximum number of new tokens")
    parser.add_argument("--preload_iter", type=int, default=1000, help="Number of preload iterations")
    parser.add_argument("--n_answers", type=int, default=1, help="Number of answers")

    args = parser.parse_args()
    if args.preload_iter==1000:
        args.preload_iter=random.randint(0,400000)
    main(args)

