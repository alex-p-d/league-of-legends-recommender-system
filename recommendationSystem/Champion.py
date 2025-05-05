"""
Author: Aleks Dimov
Email: ad3589x@gre.ac.uk

"""

from dataprocessing.DataPreprocessing import get_champ_map

class Champion():
    """Champion class referencing a champion as a separate object"""

    def __init__(self, key):
        self.key = key
        self.champ_map = get_champ_map()
        if key in self.champ_map:
            self.name = self.champ_map.get(key)
        else:
            raise ValueError('The provided key is not associated with a champion.')
        
    def __repr__(self):
        return (f'{self.key,self.name}')
    