import pandas as pd

file_path="train_embedded_light.csv"
chunksize = 200000

chunks = pd.read_csv(file_path, chunksize=chunksize)

for i, chunk in enumerate(chunks):
    chunk.to_csv(f"train_embedded_light_{i}.csv", index=False)