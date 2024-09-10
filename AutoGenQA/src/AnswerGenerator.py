import random

inst_list=[
"日本語で簡潔に回答してください",
"日本語で丁寧に回答してください ",
"日本語で回答してください",
"日本語で注意深く回答してください",
"日本語でステップ・バイ・ステップで回答してください",
"日本語でstep by stepで回答してください",
]


class AnswerGenerator:
    def __init__(self,bot,
                 max_text_len=3000,
                 n_answers=2) -> None:
        self.max_text_len=max_text_len
        self.bot=bot
        self.n_answers=n_answers

    def __call__(self, record):
        
        for i in range(self.n_answers):
            inst=random.choice(inst_list)

            if "text" in record:
                if random.randint(0,1)==1:
                    inst+="\n回答にあたり､次の文章を参考にしても良い\n"+record["text"]

            question=inst+": " +record["question"]
            question=question[:self.max_text_len]
            ans=self.bot.ask(question)
            record[f"inst_answer_{i}"]=question
            record[f"answer_{i}"]=ans[:self.max_text_len]

        return record
