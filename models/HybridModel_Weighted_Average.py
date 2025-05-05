"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

Baseline hybrid weighted average model. Developed with Content and 
Collaborative filtering on the player and champion data sets. Primarily used for
evaluation and comparison of the results with the main model.
"""

import dataprocessing.DataPreprocessing as dp
import pandas as pd
import statistics
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
import random

##########################
# CONTENT BASED FILTERING
##########################

def get_processed_df():
    raw_champ_df = dp.get_processed_data()
    # Get scaled data and drop 'key' and 'name' column since they can't be scaled
    scaled_champ_data = dp.get_scaled_data(raw_champ_df.drop(columns=['key','name']))
    # Adding the columns back
    scaled_champ_data.insert(0,'key',raw_champ_df['key']), scaled_champ_data.insert(1, 'name',raw_champ_df['name'])
    # Loading dataset
    raw_playerdf = dp.load_data('datasets/player_dataset_train_final2.csv')
    # Pre-processing dataset
    raw_playerdf = dp.preprocess_df_collaborative_filter(raw_playerdf)

    return raw_playerdf, scaled_champ_data

def group_players(raw_playerdf, scaled_champ_data):

    def merge_df(x):
        champ_data = scaled_champ_data.copy()
        x = x.rename(columns={'champion' : 'name'})
        mask = champ_data['name'].isin(x['name'])
        champ_data_filtered = champ_data[mask]
        combined_dfs = pd.merge(champ_data_filtered,x, how='left', on='name')
        mask2 = combined_dfs['mastery'] > 0.0
        combined_dfs = combined_dfs[mask2].drop(columns=['playerid','key','name'])
        return combined_dfs

    trainset_raw = raw_playerdf.groupby('playerid').head(2).groupby('playerid').apply(merge_df)
    testset_raw = raw_playerdf.groupby('playerid').tail(1).groupby('playerid').apply(merge_df)
    trainset = trainset_raw.groupby('playerid')
    testset = testset_raw.groupby('playerid')
    
    return trainset,testset

def train(trainset,testset):
    rmse_list = []
    player_list = []

    for (player_train,grp_train),(player_test,grp_test) in zip(trainset,testset):
        x_train_raw = grp_train.drop(columns='mastery')
        y_train_raw = grp_train['mastery']
        x_test_raw = grp_test.drop(columns='mastery')
        y_test_raw = grp_test['mastery']
        
        x_train,y_train,x_test,y_test = x_train_raw.to_numpy(),y_train_raw.to_numpy(),x_test_raw.to_numpy(),y_test_raw.to_numpy()
    
        model = KNeighborsRegressor(n_neighbors=2)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        player_list.append([player_test,y_test[0],pred[0]])

        rmse = root_mean_squared_error(y_test, pred)
        rmse_list.append(rmse)
    
    average_rmse = statistics.mean(rmse_list)
    print(f' CBF Test RMSE: {average_rmse}')

    return player_list

def content_filtering():
    raw_playerdf_test, scaled_champ_data = get_processed_df()
    trainset,testset = group_players(raw_playerdf_test,scaled_champ_data)
    player_list = train(trainset,testset)
    return player_list

##########################
# COLLABORATIVE FILTERING
##########################

# Load datasets
def load_collab_datasets():
    raw_df = dp.load_data('player_dataset_train_final2.csv')
    return raw_df

def preprocess_collab_datasets(raw_df):

    raw_df = dp.preprocess_df_collaborative_filter(raw_df)
    reader = Reader(rating_scale=(1,100))
    data = Dataset.load_from_df(raw_df,reader=reader)
    raw_ratings = data.raw_ratings
    random.shuffle(raw_ratings)
    threshold = int(.8 * len(raw_ratings))                                     
    trainset_raw_ratings = raw_ratings[:threshold]                             
    test_raw_ratings = raw_ratings[threshold:]                                          
    data.raw_ratings = trainset_raw_ratings  

    return trainset_raw_ratings,test_raw_ratings,data

def get_param_grid():
    param_grid = {"n_epochs": [5,10,25,50,100], 
                   "lr_all": [0.002, 0.005, 0.0075, 0.010], 
                   "reg_all": [0.4, 0.6], 
                   "n_factors": [50,75,100]}
    return param_grid

def train_model(model, param_grid, data,train_raw_ratings, test_raw_ratings):
    
    gs = GridSearchCV(model, param_grid, measures=['rmse', 'mae'], cv=5)
    gs.fit(data)

    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    algo = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Test on trainset
    testset = data.construct_testset(train_raw_ratings)
    predictions = algo.test(testset)
    print(f"{str(model)} - Accuracy on the trainset: ")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    accuracy.fcp(predictions)


    # Test on testset
    testset = data.construct_testset(test_raw_ratings)
    predictions = algo.test(testset)
    player_list_collab = []
    for index, player in enumerate(predictions):
        if (index+1) % 3 == 0:
            player_list_collab.append([player[0],player[2],player[3]])
    print(f"{str(model)} - Accuracy on the testset: ")
    accuracy.rmse(predictions)
    accuracy.mae(predictions)
    accuracy.fcp(predictions)

    return player_list_collab

def collab_filtering():
    raw_df = load_collab_datasets()
    trainset,testset,data = preprocess_collab_datasets(raw_df)
    param_grid = get_param_grid()
    player_list = train_model(SVD,param_grid,data,trainset,testset)
    return player_list

def hybrid_model():

    player_list_collaborative = collab_filtering()
    player_list_content = content_filtering()

    weight_CF = 0.6
    weight_CBF = 0.4
    hybrid_predictions = []
    real_values = []

    for i in player_list_collaborative:
        for j in player_list_content:
            if i[0] == j[0]:
                real_values.append(i[1])
                hybrid_predictions.append(i[2] * weight_CF + j[2] * weight_CBF)

    hybrid_rmse = root_mean_squared_error(real_values,hybrid_predictions)
    hybrid_mae = mean_absolute_error(real_values,hybrid_predictions)
    print(f'Hybrid RMSE: {hybrid_rmse}')
    print(f'Hybrid MAE: {hybrid_mae}')

if __name__ == '__main__':
    hybrid_model()



