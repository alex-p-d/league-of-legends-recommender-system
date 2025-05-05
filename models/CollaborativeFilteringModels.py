"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

Baseline models KNN and SVD used for Collaborative
Filtering on the player data set. Primarily used for
evaluation and comparison of the results with the main model.
"""

import dataprocessing.DataPreprocessing as dp
from surprise import SVD, Reader, Dataset, accuracy, KNNBasic
from surprise.model_selection import cross_validate, GridSearchCV, train_test_split
import random
from operator import itemgetter

def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, true_r, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[2], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


raw_df = dp.load_data('datasets/player_dataset_not_sparse.csv')
raw_df = dp.preprocess_df_collaborative_filter(raw_df)
mask_duplicated = raw_df.duplicated(subset=['playerid','champion'],keep=False)
player_data = raw_df.loc[~mask_duplicated].reset_index(drop=True)

reader = Reader(rating_scale=(0,1))
data = Dataset.load_from_df(raw_df,reader=reader)

raw_ratings = data.raw_ratings

random.shuffle(raw_ratings)

# 80% trainset, 20% testset                                                
threshold = int(.8 * len(raw_ratings))                                     
trainset_raw_ratings = raw_ratings[:threshold]                             
test_raw_ratings = raw_ratings[threshold:]                                 
                                            
data.raw_ratings = trainset_raw_ratings  

# SVD param grid
# param_grid_svd = {"n_epochs": [25], 
#                   "lr_all": [0.002, 0.005, 0.0075, 0.010], 
#                   "reg_all": [0.4, 0.6], 
#                   "n_factors": [50,75,100]}

param_grid_svd = {"n_epochs": [5,10,25,50,100], 
                  "lr_all": [0.002, 0.005, 0.0075, 0.010], 
                  "reg_all": [0.4, 0.6], 
                  "n_factors": [50,75,100]}

# KNN param grid
# param_grid_knn = {"k": [30],
#                   "min_k": [5],
#                   "sim_options":
#                                 {"name" : ["cosine"],
#                                 "user_based": [True,False], 
#                                 "min_support" : [2]},
#                   "verbose" : [True]}

param_grid_knn = {"k": [30,40,50],
                  "min_k": [1,3,5],
                  "sim_options":
                                {"name" : ["MSD","cosine","pearson","pearson_baseline"],
                                "user_based": [True,False], 
                                "min_support" : [2,3,4,5]},
                  "verbose" : [False]}

def train_model(model, param_grid, data,train_raw_ratings, test_raw_ratings):
    
    gs = GridSearchCV(model, param_grid, measures=['rmse', 'mae'], cv=5)
    gs.fit(data)

    print(gs.best_score['rmse'])
    print(gs.best_params['rmse'])

    algo = gs.best_estimator['rmse']
    trainset = data.build_full_trainset()
    algo.fit(trainset)

    # Test on trainset
    trainset_testset = data.construct_testset(train_raw_ratings)
    # predictions = algo.test(trainset_testset)
    print(f"{str(model)} - Accuracy on the trainset: ")
    # accuracy.rmse(predictions)
    # accuracy.mae(predictions)
    # accuracy.fcp(predictions)

    # Test on testset
    testset = data.construct_testset(test_raw_ratings)
    # predictions = algo.test(testset)
    # print(predictions)
    # top_n = get_top_n(predictions,n=10)
    # print(top_n)
    print(f"{str(model)} - Accuracy on the testset: ")
    
    champ_map = dp.get_champ_map()
    champ_names = [name for name in champ_map.values()]

    train_predictions = predict_for_all_champions(trainset_testset,champ_names, algo)
    test_predictions = predict_for_all_champions(testset,champ_names,algo)

    # accuracy.rmse(predictions)
    # accuracy.mae(predictions)
    # accuracy.fcp(predictions)
    return train_predictions, test_predictions


# Loops through all the champions and predicts player ratings for them
def predict_for_all_champions(dataset, champ_names, algo):
    unique_users = {}
    for user,item,rating in dataset:
        if user not in unique_users:
            unique_users[user] = {}
        unique_users[user][item] = rating
    
    predictions = []
    for user in unique_users.keys():
        for name in champ_names:
            if name in unique_users.get(user):
                mastery = unique_users.get(user).get(name)
                if mastery > 1:
                    rating = 1
            else:
                rating = 0
            predictions.append(algo.predict(user,name,r_ui=rating))
    return predictions

from collections import defaultdict

from surprise import Dataset, SVD
from surprise.model_selection import KFold

def precision_recall_at_k(predictions, k=10):
    """Return precision and recall at k metrics for each user"""
    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r > 0) for (_, true_r) in user_ratings)
        n_rec_k = len(user_ratings[:k])

        n_rel_and_rec_k = sum(
            (true_r > 0) for (est, true_r) in user_ratings[:k]
        )

        # Precision@K: Proportion of recommended items that are relevant
        # When n_rec_k is 0, Precision is undefined. We here set it to 0.

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Recall@K: Proportion of relevant items that are recommended
        # When n_rel is 0, Recall is undefined. We here set it to 0.

        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return precisions, recalls

# print("-----------------------------------------------------------------")
# train_predictions_svd, test_predictions_svd = train_model(SVD,param_grid_svd, data,trainset_raw_ratings,test_raw_ratings)
# precision_train_svd, recall_train_svd = precision_recall_at_k(train_predictions_svd,k=3)
# precision_test_svd, recall_test_svd = precision_recall_at_k(test_predictions_svd, k=3)
# avg_precision_train_svd = sum(precision_train_svd.values()) / len(precision_train_svd)
# avg_precision_test_svd = sum(precision_test_svd.values()) / len(precision_test_svd)
# avg_recall_train_svd = sum(recall_train_svd.values()) / len(recall_train_svd)
# avg_recall_test_svd = sum(recall_test_svd.values()) / len(recall_test_svd)

# print(f'SVD Average Train precision@3: {avg_precision_train_svd}, SVD Average Train recall@3: {avg_recall_train_svd}')
# print(f'SVD Average Test precision@3: {avg_precision_test_svd}, SVD Average Test recall@3: {avg_recall_test_svd}')


# print("-----------------------------------------------------------------")
# predictions_knn = train_model(KNNBasic,param_grid_knn,data,trainset_raw_ratings,test_raw_ratings)
# precision, recall = precision_recall_at_k(predictions_knn,k=3)
# avg_precision = sum(precision.values()) / len(precision)
# avg_recall = sum(recall.values()) / len(recall)
# print(f'KNN Average precision@3: {avg_precision}, KNN Average recall@3: {avg_recall}')
# print("-----------------------------------------------------------------")


train_predictions_knn, test_predictions_knn = train_model(KNNBasic,param_grid_knn, data,trainset_raw_ratings,test_raw_ratings)
precision_train_knn, recall_train_knn = precision_recall_at_k(train_predictions_knn,k=3)
precision_test_knn, recall_test_knn = precision_recall_at_k(test_predictions_knn, k=3)
avg_precision_train_knn = sum(precision_train_knn.values()) / len(precision_train_knn)
avg_precision_test_knn = sum(precision_test_knn.values()) / len(precision_test_knn)
avg_recall_train_knn = sum(recall_train_knn.values()) / len(recall_train_knn)
avg_recall_test_knn = sum(recall_test_knn.values()) / len(recall_test_knn)

print(f'KNN Average Train precision@3: {avg_precision_train_knn}, KNN Average Train recall@3: {avg_recall_train_knn}')
print(f'KNN Average Test precision@3: {avg_precision_test_knn}, KNN Average Test recall@3: {avg_recall_test_knn}')