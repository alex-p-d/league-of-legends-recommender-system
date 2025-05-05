"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

Interactions class to use as input for LightFM model.
"""

from recommendationSystem.Player import Player
from recommendationSystem.Champion import Champion
import requests
from recommendationSystem.config import API_KEY

class Interactions():
    """Mapping player interactions with champions. Used as an object to input
    in the Recommendation System class to give recommendations to players.

    It is a composite class, instantiating a Player and Champion object.

    build_interaction_list -> takes a player's top champions and mastery as input
    and returns a list in the form of [player, champion, mastery] where player
    and champion are their own separate objects.
    """
    def __init__(self,
                 player_name,
                 player_tag,
                 player_region):
        
        self.player = Player(player_name,
                             player_tag,
                             player_region)
        
    def _get_top_mastery_champions_dict(self):
        """
        Returns the player's top 3 most played champions based on mastery points.
        """

        puuid = self.player._get_puuid()
        api_url = f'https://{self.player.continent_region[0][0]}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count=15&api_key={API_KEY}'
        response = requests.get(api_url)
        champions_info = response.json()
        top_champions = {}

        for index, each in enumerate(champions_info):
                if each['championPoints'] >= 15000:
                    top_champions[champions_info[index]['championId']] = champions_info[index]['championPoints']
                else:
                    print(champions_info[index]['championId'], 'has 15,000 or less mastery, therefore not using for training.')
        return top_champions
    
    def _normalise_mastery_points(self):
        """
        Normalises the mastery points of the player's top champions on a scale between 0 and a 100.
        """

        top_champions = self._get_top_mastery_champions_dict()
        normalised_top_champions = {}

        for index, championId in enumerate(top_champions):
            if index == 0:
                highest_mastery = top_champions.get(championId)
                normalised_top_champions[championId] = 100
            if index > 0:
                normalised_top_champions[championId] = round(top_champions[championId] / highest_mastery * 100)
        
        return normalised_top_champions

    def build_interaction_list(self):
        """
        Takes player's normalised top champions and returns a list of
        player interactions in the form of [player,champion,mastery] to be used
        as input for the LightFM model.
        """
        normalised_top_champions = self._normalise_mastery_points()

        player_interactions = []

        for champion_key, mastery in normalised_top_champions.items():

            champion = Champion(str(champion_key))
            player_interactions.append([self.player.game_name+'#'+self.player.tag_name, champion.name, mastery]) 
    
        return player_interactions