import numpy as np
import pandas as pd

q1_embeddings=np.load('q1CohereEmbedLight.npy')
print(len(q1_embeddings))
embeddings_q1_df=pd.DataFrame(q1_embeddings, dtype=np.float32)
embeddings_q1_df.columns=[f'q1_emb_{i}' for i in range(384)]

q2_embeddings=np.load('q2CohereEmbedLight.npy')
print(len(q2_embeddings))
embeddings_q2_df=pd.DataFrame(q2_embeddings, dtype=np.float32)
embeddings_q2_df.columns=[f'q2_emb_{i}' for i in range(384)]

df=pd.read_csv('train.csv')
#drop all nan
df=df.dropna()
df['is_duplicate']=pd.to_numeric(df['is_duplicate'], errors='coerce')
print(df.columns)
df = pd.concat([df, embeddings_q1_df, embeddings_q2_df], axis=1)
df.drop(['question1','question2', 'id', 'qid1', 'qid2'],axis=1,inplace=True)
df.to_csv('train_embedded_light.csv',index=False)