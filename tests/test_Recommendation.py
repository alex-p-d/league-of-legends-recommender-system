import unittest
from recommendationSystem.Interactions import Interactions
from recommendationSystem.Player import Player
from recommendationSystem.Champion import Champion
from unittest.mock import patch, Mock
from recommendationSystem.config import API_KEY
from recommendationSystem.Recommendation import Recommender
import numpy as np
import models.HybridModel_FeatureCombination as fm
import pickle
import scipy as sp
from scipy import sparse

class TestRecommender(unittest.TestCase):

    def test_format_newuser_input(self):
         
         recommender = Recommender('Denji', 'Rice', 'euw1')
         
         new_user_features_list = recommender._format_newuser_input(fm.user_features_map, [recommender.player_features], recommender.num_user)
        
         dense_list = new_user_features_list.toarray()

         expected_user_features_list = np.zeros((1, 4876))
         expected_user_features_list[0,4870] = 1.0
        
         np.testing.assert_array_equal(dense_list, expected_user_features_list)

    def test_recommend(self):

        recommender = Recommender('Denji', 'Rice', 'euw1')
        recommendation_list = recommender.recommend()

        self.assertIsInstance(recommendation_list, list)
        self.assertGreaterEqual(len(recommendation_list),5)

        for recommendation in recommendation_list:
            self.assertIsInstance(recommendation, tuple)




if __name__ =='__main__':
    unittest.main()


