from .repeated_phrase import remove_repetitive_japanese
from tqdm import tqdm


def clean_text_list(text_list):
    text_list = list(set(text_list))
    cleaned_text_list = []
    for text in tqdm(text_list):
        try:
            cleaned_text = remove_repetitive_japanese(text)
        except Exception as e:
            print(e)
        if cleaned_text == "":
            continue
        cleaned_text_list.append(cleaned_text)
    return cleaned_text_list

def is_abnormal_eng_text(text, threshold=40):
    words = text.split()
    word_count = len(words)
    period_count = text.count('.')
    ratio = word_count / period_count if period_count > 0 else word_count
    #print(ratio)
    return ratio > threshold

#文字コードをもとに言語を判定
def is_japanese(text):
    japanese_char_count = 0
    total_char_count = len(text)
    
    for char in text:
        if '\u3040' <= char <= '\u309F' or '\u30A0' <= char <= '\u30FF' or '\u4E00' <= char <= '\u9FBF':
            japanese_char_count += 1
            
    return japanese_char_count / total_char_count > 0.3  # ここでは30%以上が日本語であると仮定

def clean(text,lang="ja"):
    if lang=="ja":
        #if not is_japanese(text):
        #    return ""
        #return remove_repetitive_japanese(text)
        try:
            return remove_repetitive_japanese(text)
        except:
            return ""

    elif lang=="en":
        if is_abnormal_eng_text(text):
            return ""
        return text
    else:
        raise ValueError(lang)
    
