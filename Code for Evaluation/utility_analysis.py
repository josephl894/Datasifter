"""
Used semantic similiarity and readability as metrics 
to determinet the utility of the obfuscated text
"""

#semantic similarity analysis
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_embeddings(file_path):
    df = pd.read_csv(file_path)
    arr = np.empty(0)
    for i in range(len(df)):
        row_vector = np.array(df.iloc[i])
        np.append(arr,row_vector)
    return arr

def calculate_average_similarity(original_embeddings, synthetic_embeddings):
	#cosine similarity between two sets of embeddings
    similarities = []
    for orig_emb, synth_emb in zip(original_embeddings, synthetic_embeddings):
        similarity = cosine_similarity(orig_emb.reshape(1, -1), synth_emb.reshape(1, -1))[0][0]
        similarities.append(similarity)

    return np.mean(similarities)

# Load embeddings
original_embeddings = load_embeddings('word_embeddings_original.csv')
synthetic_embeddings = load_embeddings('word_embeddings_small.csv')

# Calculate the average similarity
average_similarity = calculate_average_similarity(original_embeddings, synthetic_embeddings)
print(f"Average Cosine Similarity: {average_similarity}")

#Readability analysis
import textstat
import pandas as pd
def flesch_reading_ease_of_column(dataframe, column_name):
    return dataframe[column_name].apply(textstat.flesch_reading_ease)

df = pd.read_csv('data/mimi3.csv')
df['Readability_score'] = flesch_reading_ease_of_column(df, 'TEXT')
print("Average Readability score:", df['Readability_score'].mean())


