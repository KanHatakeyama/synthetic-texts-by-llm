import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import HfApi, logging
import glob
repo_id="kanhatakeyama/SyntheticTextCCUncleaned"
jsonl_dir = "out_data_cc/"
jsonl_list = glob.glob(f"{jsonl_dir}/*.jsonl")
jsonl_list.sort()

logging.set_verbosity_debug()
hf = HfApi()

chunk_size = 1000000  # 50万件ごとに分割

# 一時的にデータを保持するためのリスト
temp_data = []
i = 0  # チャンクのカウンター

for path in jsonl_list:
    filename = path.split("/")[-1]
    dataset_name = filename.split(".")[0]

    # JSONLファイルを読み込む
    df = pd.read_json(path, lines=True)
    
    # 一時リストにデータを追加
    temp_data.append(df)

    # 一時リストのデータを結合
    combined_df = pd.concat(temp_data, ignore_index=True)

    # チャンクサイズを超える場合、Parquetに変換してアップロード
    while len(combined_df) >= chunk_size:
        chunk = combined_df[:chunk_size]
        combined_df = combined_df[chunk_size:]
        
        table = pa.Table.from_pandas(chunk)
        parquet_path = f"{jsonl_dir}/{dataset_name}_part{i + 1}.parquet"
        pq.write_table(table, parquet_path)
        
        # Parquetファイルをアップロード
        hf.upload_file(path_or_fileobj=parquet_path,
                       path_in_repo=f"data/{dataset_name}_part{i + 1}.parquet",
                       repo_id=repo_id,
                       repo_type="dataset")
        i += 1
    
    # 処理したデータをtemp_dataから削除
    temp_data = [combined_df]

# 残りのデータもParquetに変換してアップロード
if len(combined_df) > 0:
    table = pa.Table.from_pandas(combined_df)
    parquet_path = f"{jsonl_dir}/{dataset_name}_part{i + 1}.parquet"
    pq.write_table(table, parquet_path)
    
    hf.upload_file(path_or_fileobj=parquet_path,
                   path_in_repo=f"data/{dataset_name}_part{i + 1}.parquet",
                   repo_id=repo_id,
                   repo_type="dataset")

