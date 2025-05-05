"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

Data pre-processing of the dataset provided by Riot Games.
This module is used to create the ML model of the project.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Setting display options for terminal
pd.set_option('display.max_columns', None)

# File load
def load_data(file):
    with open(file, encoding='utf8') as f:
        if file.endswith('.json'):
            df = pd.read_json(f)    
        else:
            df = pd.read_csv(f)
    return df

#################################################################################
# THE FOLLOWING FUNCTIONS ARE FOR THE CONTENT-BASED FILTERING PART OF THE MODEl.#
# PRE-PROCESSING FOR THE CHAMPIONS DATASET -- championFull.json.                #
#################################################################################

# Returns a dataframe of features to be processed for the model
def extract_champ_data(df):
    
    champ_data = df['data'].dropna()
    norm_champ_data = pd.json_normalize(champ_data)   
    data_to_process = norm_champ_data.iloc[:, np.r_[1,2,9,11,20:44,45]].copy()
    
    # The spells column is nested dictionaries in a list, so will explode it to only grab the description
    champ_spells = norm_champ_data['spells'].explode()
    spell_description = pd.json_normalize(champ_spells)[['description']]
    spell_description = spell_description.groupby(champ_spells.index)['description'].apply(' '.join)
    data_to_process = data_to_process.drop(columns='spells')
    
    # Add spell descriptions and tags to the dataframe
    data_to_process = data_to_process.assign(spell_description=spell_description)

    # Combine Spell and Passive descriptions in one column
    data_to_process['Description'] = data_to_process['spell_description'].astype(str) + " " + data_to_process['passive.description']
    data_to_process = data_to_process.drop(columns=['spell_description', 'passive.description'])

    return data_to_process

# One hot encodes a column from a dataframe  
def one_hot_encode(df, col):
    list_mask = df[col].map(lambda x: isinstance(x, list))
    if list_mask.any():
        all_tags = set(tag for sublist in df[col] for tag in sublist)
        for tag in all_tags:
            df[tag] = df[col].apply(lambda x: 1 if tag in x else 0)
        df = df.drop(col, axis=1)
    else:
        dummies = pd.get_dummies(df[[col]], dtype=int)
        df = df.drop(columns=[col])
        result = pd.concat([df, dummies], axis=1)
        return result
    return df

# Returns array of champion names
def get_champ_names(df):
    champ_names = df['name'].tolist()
    return champ_names


# Returns a list of stop words for the tfidf
def get_stop_words(champion_names):
    champion_names_lower = [i.lower() for i in champion_names]
    extra_stop_words = ['br', 'br br', 'font', 'color', 'font color']
    stop_words = list(ENGLISH_STOP_WORDS.union(champion_names_lower, extra_stop_words))
    return stop_words

# Returns the tfidf terms of given column in the dataframe
def tfidf(col, df, stop_words):
    tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,
                                        max_df=0.8,
                                        min_df=5,
                                        ngram_range=(1,4),
                                        max_features=100,
                                        lowercase=True)
    
    x = tfidf_vectorizer.fit_transform(df[col]).todense()
    tfidf_df = pd.DataFrame(x,columns=tfidf_vectorizer.get_feature_names_out())
    
    return tfidf_df

# Returns normalized numerical data
def scale_data(df):
    scaler = MinMaxScaler()
    scaled_features = df.drop(columns=df.columns[26:67])
    transformed_features = scaler.fit_transform(scaled_features.to_numpy())
    transformed_features = pd.DataFrame(transformed_features, columns=scaled_features.columns)
    df.update(transformed_features.astype(float))
    return df

def get_cosine_sim(data):
    cosine = cosine_similarity(data)
    return cosine

def get_scaled_data(data):
    scaled_features = scale_data(data)
    return scaled_features

# Get champion key:name map 
def get_champ_map():
    df = load_data('datasets/championFull.json')
    data_to_process = extract_champ_data(df)

    key_name = data_to_process[['key', 'name']].copy()
    champ_key = key_name['key'].tolist()
    champ_name = key_name['name'].tolist()

    champ_map = {}
    for key, name in zip(champ_key,champ_name):
        champ_map[key] = name
    return champ_map

def get_processed_data():
    df = load_data('datasets/championFull.json')
    data_to_process = extract_champ_data(df)    
    data_to_process = one_hot_encode(data_to_process, 'tags')
    champ_names = get_champ_names(data_to_process)
    stop_words = get_stop_words(champ_names)
    tfidf_spells = tfidf('Description', data_to_process, stop_words)
    processed_data = pd.concat([data_to_process, tfidf_spells], axis=1)
    feature_vectors = processed_data.drop(columns=['Description'])
    scaled_champions_data = get_scaled_data(feature_vectors.drop(columns=['key','name']))
    scaled_champions_data.insert(0,'name',feature_vectors['name'])
    return feature_vectors

#################################################################################
# THE FOLLOWING FUNCTIONS ARE FOR THE COLLABORATIVE FILTERING PART OF THE MODEl.#
# PRE-PROCESSING FOR THE PLAYERS DATASET -- player_dataset.csv                  #
#################################################################################

# Some analysis on the data set
def EDA(raw_df):    
    print(raw_df.describe())
    print(f'Unique values in champion column: {len(raw_df["champion"].unique())}' 
          f'\nUnique values in name column: {len(raw_df["name"].unique())}' 
          f'\nUnique values in mastery column: {len(raw_df["mastery"].unique())}')
    print(f'Null values in champion column: {raw_df["champion"].isnull().sum()}' 
          f'\nNull values in name column: {raw_df["name"].isnull().sum()}' 
          f'\nNull values in mastery column: {raw_df["mastery"].isnull().sum()}')

def preprocess_df_collaborative_filter(raw_df):
    # Normalises the masteries to be on a scale between 0 and 100
    raw_df['mastery'] = raw_df.groupby('playerid')['mastery'].transform(lambda x: round((x / x.max()) * 100))
    raw_df = raw_df.drop(columns=['region','wins','losses','veteran','freshblood','hotstreak'])
    return raw_df


#################################################################################
# THE FOLLOWING FUNCTIONS ARE FOR THE MAIN MODEL LIGHTFM.                       #
# PRE-PROCESSING FOR THE PLAYERS DATASET -- player_datase_train.csv.            #
#################################################################################

def encode_lightfm(df,col,target):
    df[col] = np.where(df[col] == target, 1, 0)
    return df

def encode_wins_losses(df, col):
    df[col] = np.where(df[col] < 50, 0,
              np.where(df[col] > 150, 2,
              np.where(df[col] > 50, 1, 
                      1)))
    return df

def preprocess_df_lightfm(raw_df):
    raw_df['mastery'] = raw_df.groupby('playerid')['mastery'].transform(lambda x: round((x / x.max()) * 100))
    raw_df = encode_lightfm(raw_df, 'veteran', True)
    raw_df = encode_lightfm(raw_df, 'freshblood', True)
    raw_df = encode_lightfm(raw_df, 'hotstreak', True)
    raw_df = encode_wins_losses(raw_df, 'wins')
    raw_df = encode_wins_losses(raw_df, 'losses')
    raw_df = one_hot_encode(raw_df, 'region')    
    return raw_df


if __name__ == '__main__':
    df = load_data('datasets/championFull.json')
    print(df.head())