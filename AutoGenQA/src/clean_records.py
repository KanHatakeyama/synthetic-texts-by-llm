

noise_texts="""
### 新しい指示
**問題**：
**問題**
**指示**:
**質問:** 
[問題]
【問題】
【指示】
【新しい問題】
【新指示】
【新しい指示】
【新規指示】
【質問】
問題：
類題：
（指示）
:
:
問題
指示
文章：
新しい指示
"""

noise_list=noise_texts.split("\n")
noise_list=[n for n in noise_list if n]

def clean_question(text:str):
    if text is None:
        return ""
    for noise in noise_list:
        text=text.replace(noise,"")

    text=text.strip()
    if text.startswith("\"") and text.endswith("\""):
        text=text[1:-1]

    text=text.strip()
    return text
