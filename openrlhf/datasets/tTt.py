import pandas as pd

# 读取 parquet 文件
df = pd.read_parquet('train.parquet')

# 转换为 JSON 格式
df.to_json('train.json', orient='records', lines=True)
