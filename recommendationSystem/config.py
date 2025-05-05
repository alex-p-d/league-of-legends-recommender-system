# Configurations file

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEY')

account_region = {'europe' : ['euw1', 'eun1'],
                  'americas' : ['na1', 'la1', 'la2', 'br1'],
                  'asia' : ['kr1', 'jp1']}


