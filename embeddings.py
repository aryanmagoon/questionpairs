import pandas as pd
import numpy as np
import cohere

df=pd.read_csv('train.csv')
#drop all nan
df=df.dropna()
q1arr=[x for x in df['question1'].values]
q2arr=[x for x in df['question2'].values]
print(len(q1arr))

def split_array_into_batches(arr, batch_size):
    batches = []
    for i in range(0, len(arr), batch_size):
        batches.append(arr[i:i + batch_size])
    return batches

arrs=split_array_into_batches(q1arr,1500)
q2arrs=split_array_into_batches(q2arr,1500)

co = cohere.Client('xxx') # This is your trial API key

q1_embeddings = []  # List to store embeddings from all batches

for i in range(len(arrs)):
    print(f"Processing batch {i}")
    response = co.embed(
        model='embed-english-light-v3.0',
        texts=arrs[i],
        input_type='classification'
    )
    embeddings = response.embeddings

    # Add the embeddings of the current batch to the list
    if i==0:
        q1_embeddings=np.array(embeddings)
    else:
        q1_embeddings = np.append(q1_embeddings, embeddings, axis=0)

    # Save the updated list of embeddings to the pickle file
    if i % 30 == 0:
        np.save('q1CohereEmbedLight', q1_embeddings)
np.save('q1CohereEmbedLight', q1_embeddings)
q2_embeddings = []  # List to store embeddings from all batches
for i in range(len(q2arrs)):
    print(f"Processing batch {i}")
    response = co.embed(
        model='embed-english-light-v3.0',
        texts=q2arrs[i],
        input_type='classification'
    )
    embeddings = response.embeddings

    if i==0:
        q2_embeddings=np.array(embeddings)
    else:
        q2_embeddings = np.append(q2_embeddings, embeddings, axis=0)

    # Save the updated list of embeddings to the pickle file
    if i % 30 == 0:
        np.save('q2CohereEmbedLight', q2_embeddings)
np.save('q2CohereEmbedLight', q2_embeddings)
