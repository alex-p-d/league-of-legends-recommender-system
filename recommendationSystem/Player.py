"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

This module contains a Player class which stores
a player's id, tag, most played champions, and
the mastery points of said champions.

"""

import requests
from recommendationSystem.config import API_KEY, account_region

class Player():
    def __init__(self, game_name, tag_name, region):
        self.game_name = game_name
        self.tag_name = tag_name
        self.region = region
        self.continent_region = self._get_continent_region(self.region)

    def _get_continent_region(self, region):
        """
        Returns a list of continent and region for the player.
        """
        continent_region = [[region, continent] for continent in account_region if region in account_region.get(continent)]
        return continent_region
   
    def _get_puuid(self):
        """
        Returns the puuid of the player.
        """
            
        api_url = f'https://{self.continent_region[0][1]}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{self.game_name}/{self.tag_name}?api_key={API_KEY}'
        response = requests.get(api_url)
            
        if response.status_code == 404:
            raise ValueError('Player not found in Riot\'s API, please try again with another IGN.')
        if response.status_code != 200:
            print(response.status_code)
            raise ValueError('Something went wrong with the API..')

        player_info = response.json()
        puuid = player_info["puuid"]

        return puuid
    
    def get_account_info(self):
        puuid = self._get_puuid()

        api_url = f'https://{self.continent_region[0][0]}.api.riotgames.com/lol/summoner/v4/summoners/by-puuid/{puuid}?api_key={API_KEY}'
        response = requests.get(api_url)
        account_info = response.json()

        account_info_dict = {}

        for key in account_info.keys():
            if key == 'profileIconId':
                account_info_dict.update({key:account_info.get(key)})
            if key == 'summonerLevel':
                account_info_dict.update({key:account_info.get(key)})

        return account_info_dict       

    def get_player_features(self):
        """
        Takes the player features from the Riot Games API and returns an iterable
        of the form [player, {feature : rating}] to be used as input for the
        LightFM model. 
        """

        puuid = self._get_puuid()
        api_url = f'https://{self.continent_region[0][0]}.api.riotgames.com/lol/league/v4/entries/by-puuid/{puuid}?api_key={API_KEY}'
        response = requests.get(api_url)
        player_info = response.json()
  
        def _normalise_numerical(feature):
            if feature < 50:
                return 0
            elif feature > 150:
                return 2
            else:
                return 1
            
        def _normalise_category(feature):
            return 1 if feature == True else 0 
                
        if player_info != None and len(player_info) > 0:
            player_features_tuple = (self.game_name+'#'+self.tag_name,
                                    {'region_'+self.region : 1,
                                    'wins': _normalise_numerical(player_info[0].get('wins')),
                                    'losses': _normalise_numerical(player_info[0].get('losses')),
                                    'veteran': _normalise_category(player_info[0].get('veteran')),
                                    'freshBlood': _normalise_category(player_info[0].get('freshBlood')),
                                    'hotStreak': _normalise_category(player_info[0].get('hotStreak'))})
            return player_features_tuple
        else:
            print('Player hasn\'t played recently, so only region will be set as a feature.')
            player_features_tuple = (self.game_name+'#'+self.tag_name,
                                    {'region_'+self.region : 1})
            return player_features_tuple
