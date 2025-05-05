import numpy as np
import models.HybridModel_FeatureCombination as fm
import pickle
import scipy as sp
from scipy import sparse
from recommendationSystem.Player import Player
from recommendationSystem.Interactions import Interactions


class Recommender():
    """
    Recommender class is the object which recommends champions
    to players. Takes player,interactions and champion objects and passes them
    to the fit_partial function of the LightFM model to generate top-N recommendations
    for that particular user. 
    """
    def __init__(self, playerid, tag, region):
        self.player = Player(playerid,tag,region)
        self.interactions_object = Interactions(playerid,tag,region)
        self.player_features = self.player.get_player_features()
        self.interactions_list = self.interactions_object.build_interaction_list()
        self.num_user = len(fm.user_map)
        self.num_items = fm.interactions.shape[1]
        self.all_keys = list(fm.user_features_map.keys())
        self.feature_keys = self.all_keys[self.num_user:]
        with open('serialised_models/best_model_newDF_withoutFeatures_FINAL.pickle', 'rb') as f:
            """Loads pickled LightFM model"""
            self.model = pickle.load(f)
        self._fit_new_user()
        
    def _format_newuser_input(self,user_features_map,user_features_list,num_user):
            
        normalised_val = 1.0
        target_indices = []

        all_keys = list(user_features_map.keys())
        feature_keys = set(all_keys[num_user:])

        for key,value in user_features_list[0][1].items():
            if value > 0:
                if key in feature_keys:
                    try:
                        target_indices.append(user_features_map[key])
                    except KeyError:
                        print("New user feature encountered '{}'".format(key))
                        pass

        new_user_features = np.zeros(len(user_features_map.keys()))
        for i in target_indices:
            new_user_features[i] = normalised_val
        new_user_features = sparse.csr_matrix(new_user_features)
        return new_user_features

                    
    def _expand_new_user_matrix(self):
        new_user_features_list = self._format_newuser_input(fm.user_features_map,[self.player_features],self.num_user)

        new_row = np.zeros(self.num_items)

        for interaction in self.interactions_list:
            champion_name = interaction[1]
            mastery = interaction[2]
            if champion_name in fm.item_map:
                item_index = fm.item_map[champion_name]
                new_row[item_index] = mastery

        new_sparse_row  = sparse.csr_matrix(new_row)
        new_matrix = sp.sparse.vstack([fm.interactions, new_sparse_row], format='csr')

        new_user_features_matrix = sp.sparse.vstack([fm.user_features, new_user_features_list], format='csr')
        
        return new_matrix, new_user_features_matrix

    def _fit_new_user(self):
        new_matrix, new_user_features_matrix = self._expand_new_user_matrix()
        if self.player_features[0] not in fm.user_map:
            print('Fitting new player please wait for a moment...')
            self.model.fit_partial(new_matrix, item_features=fm.item_features, epochs=1)
        else:
            print('Player was already in the training set so he will not be fit again.')
            

    def recommend(self):
        new_matrix, new_user_features_matrix = self._expand_new_user_matrix()
        new_player_index = new_matrix.shape[0] - 1
        existing_user_index = fm.user_map.get(self.player_features[0])
        items = list(range(0,170))

        try: 
            if self.player_features[0] not in fm.user_map:
                print('Recommending for new user...')
                pred = self.model.predict(new_player_index, items, item_features=fm.item_features)   
            else:
                print('Recommending for user that was in the training set.')
                pred = self.model.predict(existing_user_index, items, item_features=fm.item_features)

            top10_indices = np.argsort(pred)[-6:][::-1]
            item_map_inv = {v: k for k, v in fm.item_map.items()}
            player_champions_list = [player[1] for player in self.interactions_list]
            counter=0
            print(f"Top 6 recommended champions for the user {self.player.game_name + '#' + self.player.tag_name} are:")
            recommendation_list = []
            for idx in top10_indices:
                champion_name = item_map_inv.get(idx, "Unknown")
                if champion_name not in player_champions_list and counter <= 2:
                        score = pred[idx]
                        print(f"Champion: {champion_name}, Score: {score}")
                        recommendation_list.append((champion_name,float(score)))
                        counter+=1
            return recommendation_list
        except Exception as e:
            print(e)


