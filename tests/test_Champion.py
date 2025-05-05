import unittest
from recommendationSystem.Champion import Champion

class TestChampion(unittest.TestCase):

    def test__init__(self):

        champion = Champion('1')
        expected_champion_name = 'Annie'
        self.assertEqual(expected_champion_name, champion.name)


if __name__ =='__main__':
    unittest.main()
