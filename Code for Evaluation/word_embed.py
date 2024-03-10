from gensim.models import Word2Vec
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np

wv = api.load('word2vec-google-news-300')

def get_word_embeddings(text, model):
    tokens = word_tokenize(text)
    embeddings = [model[word] for word in tokens if word in model]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(model.vector_size)

file_path_original = 'data/mimi3.csv'
file_path_small = 'data/unstructured_obfuscated/dt_sm.csv'
df_original = pd.read_csv(file_path_original)
df_small = pd.read_csv(file_path_small)

df_original['word_embeddings'] = df_original['TEXT'].apply(lambda text: get_word_embeddings(text, wv))
df_small['word_embeddings'] = df_small['TEXT'].apply(lambda text: get_word_embeddings(text, wv))

embeddings_df_original = pd.DataFrame(df_original['word_embeddings'].tolist())
embeddings_df_small = pd.DataFrame(df_small['word_embeddings'].tolist())

embeddings_df_original.to_csv('word_embeddings_original.csv', index=False)
embeddings_df_small.to_csv('word_embeddings_small.csv', index=False)