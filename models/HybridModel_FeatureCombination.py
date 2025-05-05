"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

Main model developed with usage of LightFM (Hybrid Recommendation Model).
Combines both attributes from the content-based filtering with
user-item data from the collaborative filtering to provide champion
predictions to players. The model has been trained on Google Collab as
the loss function doesn't compile properlty on Windows. It was has been
pickled and then loaded to predict/fit_partial. 
"""

from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.evaluation import recall_at_k
from lightfm.cross_validation import random_train_test_split
from dataprocessing.DataPreprocessing import get_processed_data, get_scaled_data, load_data, preprocess_df_lightfm
import itertools
import numpy as np
from LightFMResizable import LightFMResizable
import pickle

champions_data = get_processed_data()
scaled_champions_data = get_scaled_data(champions_data.drop(columns=['key','name']))
scaled_champions_data.insert(0,'name',champions_data['name'])

player_data_raw = load_data('datasets/player_dataset_not_sparse.csv')
player_data = preprocess_df_lightfm(player_data_raw)

mask_duplicated = player_data.duplicated(subset=['playerid','champion'],keep=False)
player_data = player_data.loc[~mask_duplicated].reset_index(drop=True)

def get_user_item_features(player_data,champion_data):
   
    champion_data_dict = champion_data.to_dict(orient='records')
    item_features = [[i.pop('name'),i] for i in champion_data_dict]
    
    # Drop user features we dont want
    player_features_df = player_data.drop(columns=['champion','mastery','veteran'])
    #['champion','mastery','veteran', 'wins', 'losses', 'freshblood','hotstreak']
    player_features_dict = player_features_df.to_dict(orient='records')
    user_features = [[i.pop('playerid'),i] for i in player_features_dict]

    return item_features, user_features

dataset = Dataset()

dataset.fit(users=tuple(player_data['playerid'].unique()),
            items=tuple(champions_data['name']),
            item_features=tuple(champions_data.columns.drop(['key','name'])),
            user_features=tuple(player_data.columns.drop(['playerid','champion','mastery','veteran'])))

#user_features=tuple(player_data.columns.drop(['playerid','champion','mastery','veteran']))


interactions, weights = dataset.build_interactions(
player_data[['playerid', 'champion', 'mastery']].values.tolist())

train_interactions, test_interactions = random_train_test_split(interactions,
                                                                test_percentage=0.2,
                                                                random_state=np.random.RandomState(42))

train_weights, test_weights = random_train_test_split(weights, test_percentage=0.2,random_state=np.random.RandomState(42))

test_interactions, val_interactions = random_train_test_split(test_interactions,
                                                                test_percentage=0.5,
                                                                random_state=np.random.RandomState(42))

test_weights, val_weights = random_train_test_split(test_weights, test_percentage=0.5,random_state=np.random.RandomState(42))

item_features_list, user_features_list = get_user_item_features(player_data,scaled_champions_data)
item_features = dataset.build_item_features(item_features_list)
user_features = dataset.build_user_features(user_features_list)

user_map, user_features_map, item_map, item_features_map = dataset.mapping()


def sample_hyperparameters():

  while True:
    yield {
          'no_components': np.random.randint(5,32),
          'learning_schedule': np.random.choice(['adagrad','adadelta']),
          'loss': np.random.choice(['bpr','warp']),
          'learning_rate': np.random.exponential(0.05),
          'item_alpha': np.random.exponential(1e-6),
          'user_alpha': np.random.exponential(1e-6),
          'max_sampled': np.random.randint(5,50),
          'num_epochs': np.random.randint(5,100)
          }

def random_search(train, val, test, item_features, user_features, num_samples=30, num_threads=2):

    best_score = 0

    for hyperparams in itertools.islice(sample_hyperparameters(), num_samples):
        num_epochs = hyperparams.pop('num_epochs')

        model = LightFMResizable(**hyperparams)
           
        model.fit(interactions=train,
                  item_features=item_features,
                  user_features=user_features,
                  sample_weight=train_weights,
                  epochs=num_epochs,
                  num_threads=num_threads,
                  verbose=True
                  )

        train_precision = precision_at_k(model, train,item_features=item_features,user_features=user_features, k=3).mean()
        train_auc = auc_score(model, train, item_features=item_features,user_features=user_features).mean()
        train_recall = recall_at_k(model, train, item_features=item_features,user_features=user_features,k=3).mean()
        val_precision = precision_at_k(model, val,train_interactions=train,item_features=item_features,user_features=user_features, k=3).mean()
        print(f'Train Recall: {train_recall:.4f}, Train AUC: {train_auc:.4f}')
        print(f"Train Precision: {train_precision:.4f}, Validation Precision: {val_precision:.4f}")

        if val_precision > best_score:
            best_score = val_precision
            best_params = hyperparams
            best_epoch = num_epochs
            best_model = model

    print(f"Best Precision: {best_score:.4f},Best Epoch: {best_epoch} ,Best Parameters: {best_params}")
    test_precision = precision_at_k(best_model, test,train_interactions=train,item_features=item_features,user_features=user_features, k=3).mean()
    test_auc = auc_score(best_model,test,train_interactions=train,item_features=item_features,user_features=user_features).mean()
    test_recall = recall_at_k(best_model,test,train_interactions=train,item_features=item_features,user_features=user_features,k=3).mean()

    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test AUC: {test_auc:.4f}")
    print(f"Test Recall: {test_recall:.4f}")

    return best_model, best_params, best_score

# # Gets best model and hyperparams
# best_model, best_params, best_score = random_search(train_interactions, val_interactions, test_interactions, item_features, user_features)

# with open('best_model_newDF_withFeatures_FINAL.pickle', 'wb') as f:
#    pickle.dump(best_model, f, protocol=pickle.HIGHEST_PROTOCOL)

