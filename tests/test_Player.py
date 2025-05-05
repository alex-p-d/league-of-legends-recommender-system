import unittest
from recommendationSystem.Player import Player
from unittest.mock import patch, Mock
from recommendationSystem.config import API_KEY

class TestPlayer(unittest.TestCase):

    @patch('recommendationSystem.Player.requests.get')
    def test_get_puuid(self, mock_get):
         
         mock_response = Mock()
         mock_response.status_code = 200
         player_info = {'puuid': 'ywGMe79XxLYi2AIirijvHFD4Kcl35bhHLRgJUD0SgbXCLTN6FFKDuycd15y7CeF4-8guzzUBKH05zQ', 'gameName': 'Denji', 'tagLine': 'Rice'}
         mock_response.json.return_value = player_info

         mock_get.return_value = mock_response

         player = Player('Denji','Rice','euw1')
         player_puuid = player._get_puuid()
         mock_get.assert_called_with(f'https://{player.continent_region[0][1]}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{player.game_name}/{player.tag_name}?api_key={API_KEY}')
         
         self.assertEqual(player_puuid, player_info['puuid'])

    @patch('recommendationSystem.Player.requests.get')
    @patch.object(Player, '_get_puuid')
    def test_get_player_features(self, mock_puuid,mock_get):

        mock_response = Mock()
        mock_puuid.return_value = 'RGten7QbHVBpW6WK9V5AMl3gbXBwRcBEJhB2b4rwm7fHAOqk8N67qBPfoxpbjP7NhaFyp5N_lN2LQQ'
        player_info = [{'leagueId': '79b329f5-cc98-3716-95f4-463e7e6e80a0', 'queueType': 'RANKED_SOLO_5x5', 'tier': 'CHALLENGER', 'rank': 'I', 'summonerId': '68K7WfmFaOR5xCPWG7mmjrAgJ266le1jysjKRpyyVfSTnAFFJSwuMNAf3w', 'puuid': 'RGten7QbHVBpW6WK9V5AMl3gbXBwRcBEJhB2b4rwm7fHAOqk8N67qBPfoxpbjP7NhaFyp5N_lN2LQQ', 'leaguePoints': 1946, 'wins': 218, 'losses': 120, 'veteran': True, 'inactive': False, 'freshBlood': False, 'hotStreak': True}]
        mock_response.json.return_value = player_info
        mock_get.return_value = mock_response

        player = Player('Naak Pado', 'VIT', 'euw1')
        player_features = player.get_player_features()
        mock_get.assert_called_with(f'https://{player.continent_region[0][0]}.api.riotgames.com/lol/league/v4/entries/by-puuid/{mock_puuid.return_value}?api_key={API_KEY}')

        expected_player_features = ('Naak Pado#VIT', {'region_euw1': 1, 'wins': 2, 'losses': 1, 'veteran': 1, 'freshBlood': 0, 'hotStreak': 1})

        self.assertEqual(player_features, expected_player_features)


if __name__ =='__main__':
    unittest.main()

