import os
import subprocess

# リポジトリURLのリスト (privateを含む)

dateset_texts = """
https://huggingface.co/datasets/kanhatakeyama/wizardlm8x22b-logical-math-coding-sft_additional
https://huggingface.co/datasets/kanhatakeyama/wizardlm8x22b-logical-math-coding-sft_additional-ja
https://huggingface.co/datasets/kanhatakeyama/wizardlm8x22b-logical-math-coding-sft-ja
https://huggingface.co/datasets/kanhatakeyama/multiturn-Calm3-manual
https://huggingface.co/datasets/kanhatakeyama/wizardlm8x22b-logical-math-coding-sft
https://huggingface.co/datasets/kanhatakeyama/ramdom-to-fixed-multiturn-Calm3
https://huggingface.co/datasets/kanhatakeyama/logical-wizardlm-7b-ja-0805
https://huggingface.co/datasets/kanhatakeyama/0804calm3-logical-multiturn-pretrain
https://huggingface.co/datasets/kanhatakeyama/logical-wizardlm-7b-ja-0731
https://huggingface.co/datasets/kanhatakeyama/logical-wizardlm-7b-ja-0730
https://huggingface.co/datasets/kanhatakeyama/logical-wizardlm-7b
https://huggingface.co/datasets/kanhatakeyama/logical-wizardlm-7b-ja
https://huggingface.co/datasets/kanhatakeyama/0723-calm3-22b-random-genre-inst-sft-multiturn-clean-tsub
https://huggingface.co/datasets/kanhatakeyama/logicaltext-wizardlm8x22b-Ja
https://huggingface.co/datasets/kanhatakeyama/logicaltext-wizardlm8x22b-api
https://huggingface.co/datasets/kanhatakeyama/0722-calm3-22b-random-genre-inst-sft-multiturn-tsub
https://huggingface.co/datasets/kanhatakeyama/SyntheticTextCCUncleaned
https://huggingface.co/datasets/kanhatakeyama/0717-calm3-22b-random-genre-inst-sft-tsub
https://huggingface.co/datasets/kanhatakeyama/0719-calm3-22b-random-genre-inst-sft-multiturn-tsub
https://huggingface.co/datasets/kanhatakeyama/logicaltext-wizardlm8x22b
https://huggingface.co/datasets/kanhatakeyama/0717-calm3-22b-random-genre-inst-sft-tsub-part
https://huggingface.co/datasets/kanhatakeyama/AutoMultiTurnByCalm3-22B
https://huggingface.co/datasets/kanhatakeyama/CommonCrawl-RAG-QA-Calm3-22b-chat
https://huggingface.co/datasets/kanhatakeyama/SyntheticTextWikiTranslate
https://huggingface.co/datasets/kanhatakeyama/SyntheticText
https://huggingface.co/datasets/kanhatakeyama/SyntheticTextCC
https://huggingface.co/datasets/kanhatakeyama/CreativeCommons-RAG-QA-Mixtral8x22b
https://huggingface.co/datasets/kanhatakeyama/databricks-dolly-15k-ja-regen-nemotron
https://huggingface.co/datasets/kanhatakeyama/OpenMathInstruct-ja-phi3
https://huggingface.co/datasets/kanhatakeyama/SyntheticTextOpenMathInstruct
https://huggingface.co/datasets/kanhatakeyama/LogicalDatasetsByMixtral8x22b
https://huggingface.co/datasets/kanhatakeyama/AutoMultiTurnByMixtral8x22b
https://huggingface.co/datasets/kanhatakeyama/OrcaJaMixtral8x22b
https://huggingface.co/datasets/kanhatakeyama/ChatbotArenaJaMixtral8x22b
https://huggingface.co/datasets/kanhatakeyama/AutoWikiQA
https://huggingface.co/datasets/kanhatakeyama/0804ramdom-to-fixed-multiturn-Calm3-pretrain-tsub
https://huggingface.co/datasets/team-hatakeyama-phase2/Synthetic-JP-EN-Coding-Dataset-453k
https://huggingface.co/datasets/team-hatakeyama-phase2/Open-Platypus-Japanese
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-rule-arithmetic-qa
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-topic-jp-reasoning-nemotron-4
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-topic-jp-reasoning-calm3
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-topic-jp-basic-reasoning-calm3
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-topic-jp-basic-math-calm3
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-topic-jp-highschoolmath-nemotron-4
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-persona-jp-math-nemotron-4
https://huggingface.co/datasets/weblab-GENIAC/phase2-synth-topic-jp-coding-calm3
https://huggingface.co/datasets/weblab-GENIAC/phase2-calm-generated-texts-commoncrawl
"""

dataset_urls = dateset_texts.split("\n")
dataset_urls = [i for i in dataset_urls if i != ""]


# データセットを保存するディレクトリ
output_dir = "datasets"
os.makedirs(output_dir, exist_ok=True)

for url in dataset_urls:
    # リポジトリ名を抽出してディレクトリ名として利用
    repo_name = url.split("/")[-1]
    repo_dir = os.path.join(output_dir, repo_name)

    # クローン（空の状態）
    subprocess.run(["git", "clone", "--no-checkout", url, repo_dir])

    # 指定ディレクトリに移動
    os.chdir(repo_dir)

    # 空のコミット状態を設定
    subprocess.run(["git", "reset", "--hard", "HEAD"])

    # `jsonl` と `parquet` のみダウンロード
    subprocess.run(["git", "lfs", "pull", "--include", "*.jsonl,*.parquet"])

    # 元のディレクトリに戻る
    os.chdir("../../")

    print(f"Downloaded: {repo_name}")

print("Completed downloading specified files.")

