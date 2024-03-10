#Privacy Evaluation
#performed identity disclosure analysis (according to paper)
#Identity Disclosure Analysis using spaCy's NER (Named Entity Recognition) model
import spacy
import pandas as pd

def identify_sensitive_entities(text, nlp_model):
    """
    Returns: a list of identified sensitive entities.
    """
    doc = nlp_model(text)
    sensitive_entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'DATE', 'ORG', 'GPE', 'CARDINAL', 'TIME']]
    print(sensitive_entities,'\n')
    return sensitive_entities

def identify_disclosure_analysis(df, nlp_model):
    """
    Returns: df with identified sensitive entities for each row.
    """
    df['sensitive_entities'] = df['TEXT'].apply(lambda text: identify_sensitive_entities(text, nlp_model))
    return df

def calculate_document_privacy_score(entities, entity_weights):
    """
    Calculate the privacy score for a single document based on identified entities.

    Args:
    entities (list): list of entities identified in the document
    entity_weights (dict): keys are entity types, weights are values

    Returns:
    float: document privacy score
    """
    score = sum(entity_weights.get(ent, 0) for ent in entities)
    return score

def calculate_overall_privacy_score(df, entity_column, entity_weights):
    """
    Calculate the overall privacy score for the dataset.

    Args:
    df : analysis results
    entity_column (str): name of the column containing the identified entities
    entity_weights (dict): keys are entity types, values are weights

    Returns:
    float: overall privacy score for the dataset.
    """
    total_score = df[entity_column].apply(lambda entities: calculate_document_privacy_score(entities, entity_weights)).sum()
    overall_score = total_score / len(df)
    return overall_score

# Define weights for each entity type (adjust as needed)
entity_weights = {
    'PERSON': 10,
    'DATE': 5,
    'ORG': 3,
    'GPE': 2,
    'CARDINAL': 2,
    'TIME': 2
}
nlp = spacy.load("en_core_web_sm")
file_path_original = 'data/mimi3.csv'
file_path_small = 'data/unstructured_obfuscated/dt_sm.csv'
df_original = pd.read_csv(file_path_original)
df_small = pd.read_csv(file_path_small)

sensitive_data_analysis_original = identify_disclosure_analysis(df_original, nlp)
sensitive_data_analysis_small = identify_disclosure_analysis(df_small, nlp)


overall_privacy_score_original = calculate_overall_privacy_score(sensitive_data_analysis_original, 'sensitive_entities', entity_weights)
overall_privacy_score_obfuscated = calculate_overall_privacy_score(sensitive_data_analysis_small, 'sensitive_entities', entity_weights)
print(f"Overall Privacy Score for original text: {overall_privacy_score_original}")
print(f"Overall Privacy Score for obfuscated text: {overall_privacy_score_obfuscated}")
