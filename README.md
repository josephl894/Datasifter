**Work in Progress

DataSifter, a data obfuscation algorithm for senstitive clinical data. 

Latest Update: Dec 2023**

# Steps 1-6


## Step 1 - Preprocessing

For the preprocessing step of this algorithm, we performed the following common preprocessing techniques:

1. Expand contractions
2. Lemmatization
3. Stemming
4. Converted all characters to lowercase
5. Removed non-alphanumeric characters




## Step 2 - Identify Sensitive outcomes to protect

1. 'Length_of_stay_avg’: continuous
2. 'Religion': categorical with 5 levels
3. 'Gender': binary

These are the three sensitive outcomes that we wanted to protect.



## Step 3 - Using LightGBM to identify keywords to protect in the text data

1. A dictionary of keywords is initialized to store the results of feature importance for different outcomes. It then iterates over various outcomes.
2. For each outcome, it processes the data: Extracts the target variable y from the dataframe df. Converts the text column TEXT into a numerical format using CountVectorizer using unigrams.
3. The data is split into training and test sets using train_test_split. This is in LightGBM-specific data format using lgb.Dataset.
4. Hyperparameters for the LightGBM model are defined: num_leaves: 31, metric: 'multi_logloss', num_round (#boosting rounds): 10
5. The LightGBM model is trained using the lgb.train function with the defined hyperparameters. Validation is then performed on the test set.
6. The top predicative keywords for each sensitive outcome are stored in the keywords dictionary.



## Step 4 - Build Semantic radius around the top keywords using word2vec and generate keywords to replace based on obfuscation leve

For example, the top 10 keywords for the outcome “Marital Status” are: 'wife', 'husband', 'married', 'alone', 'daughter', 'widowed', 'son', 'lives', 'sex', 'she'. Based on semantic meanings, we can extract two “semantic clusters”: 1: 'wife', 'husband', 'daughter', ‘son’; 2: 'married', 'alone', 'widowed'. This step is to guarantee the readability of obfuscated text.

Input 1 (for obfuscation): # keywords (higher = more obfuscation)

Input 2  (for obfuscation): radius around each keyword using word2vec (further away = more obfuscation) 

1. A class MySentences is defined to process the text data, which splits each sentence into words and yields them as separate tokens. It's designed to work with the Word2Vec model, which requires tokenized sentences as input.
2. A class W2V is defined to encapsulate the Word2Vec model training and querying processes.
- The constructor of this class initializes the Word2Vec model with various hyperparameters like min_count, window, vector_size, and others.
- get_similar_word() method queries the model for the most similar words to each keyword, using the most_similar function of the Word2Vec model. It takes a list of keywords and a specified semantic radius range (start_radius and end_radius). It returns a random word from the list of words that are semantically similar to each keyword within the specified radius.
- obfuscate() function replaces keywords in a sentence with similar words obtained from the model
3. We then obtain the top 10 keywords for each sensitive outcome based on the obfuscation level. We store this in a vector called wordsReplacement


## Step 5 - Apply obfuscation to text
Taking in sensitive factors (outcomes), level of obfuscation, wordForReplacement dictionary as input, the obfuscate() method replaces the sensitive words with their corresponding keywords based on the level of obfuscation. This obfuscation is applied three times to yield three different datasets, once for each level of obfuscation of small, medium, and large.

wordForReplacement: A dictionary where keys are sensitive factors (outcomes) and values are DataFrames containing words and their corresponding replacements at different obfuscation levels. This is generated from the previous step as the vector wordsReplacement.


## Step 6 - Evaluate based on utility and privacy metric
Recast Original text and Obfuscated text into tabular format

### Utility analysis
**Semantic similarity**
- Original and obfuscated text is cast into word embedding using word2vec. The word2vec model is trained on the ‘TEXT’ column of the MIMIC-III dataset. 
= A similarity score calculated for each text entry using cosine similarity and the average is taken, producing a similarity score. Higher score means more similarity between the texts.

**Readability analysis**
- Returns a score based on the flesch_reading_ease metric
= Makes sure that the obfuscated text data is coherent and interpretable
= Run the readability analysis on the original text to obtain baseline readability (~52.3295). Then run readability analysis on the obfuscated texts to obtain scores and compare.

Training a classification model (with BERT) using the original text and apply this model to the sifted text to calculate data utility loss (For future implementation)

### Privacy analysis
**Identity disclosure risk analysis**

Using spaCy's NER (Named Entity Recognition) model, find the sensitive entities, which could be used to identify the patient, that exists in the obfuscated data.

The sensitive labels are: 
1. PERSON: Names of patients, relatives, or doctors.
2. DATE: Dates of appointments, procedures, births, etc.
3. ORG: Names of hospitals, insurance companies, or other organizations.
4. GPE: Geographical locations like cities, countries, which might indicate where the patient lives or where the treatment occurred.
5. CARDINAL: Numbers that could include things like patient IDs, room numbers, or other identifying numeric data.
6. TIME: Specific times that might be related to appointments or procedures.

A privacy score is calculated based on the relative weighting of the sensitive entities detected in the obfuscated data
Weighting is assigned as follows:
1. 'PERSON': 10
2. 'DATE': 5
3. 'ORG': 3
4. 'GPE': 2
5. 'CARDINAL': 2
6. 'TIME': 2

Compute a privacy score for the original dataset first. Then compute a privacy score for each of the small, medium, and large obfuscated datasets. Finally, compare the scores relative to each other to find the best privacy-preserving obfuscation parameters.


Compute BLEU score between text entries. (For future implementation)
