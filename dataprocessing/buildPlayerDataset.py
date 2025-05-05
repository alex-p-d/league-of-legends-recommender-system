import requests
from recommendationSystem.config import API_KEY
import time
import dataprocessing.DataPreprocessing as dp
import pandas as pd
import csv

# regions = [{'europe' : 'euw1'},{'asia' : 'kr'}]
# regions1=[{'europe':'eun1'}]
# tiers1=['BRONZE']

tiers = ['GOLD','PLATINUM','EMERALD']

regions = [{'europe' : 'eun1'},
            {'europe' : 'euw1'},
            {'americas' : 'na1'},
            {'americas' : 'la1'},
            {'americas' : 'la2'},
            {'americas' : 'br1'},
            {'asia' : 'kr'},
            {'asia' : 'jp1'} ]

def get_continent(x,regions):
    for dict in regions:
        for continent, region in dict.items():
            if x == dict.get(continent):
                return continent
            
def get_champ_map():

    champion_data = dp.get_processed_data()
    df_champion_data = pd.DataFrame(champion_data)
    key_name = df_champion_data[['key', 'name']].copy()
    champ_key = key_name['key'].tolist()
    champ_name = key_name['name'].tolist()

    champ_map = {}
    for key, name in zip(champ_key,champ_name):
        champ_map[key] = name
    return champ_map

def get_player_info(tiers, regions):

    player_features = []
    puuids = []
    for tier in tiers:
        for region in regions:
            for each, continent in region.items():            
                api_url = f'https://{region.get(each)}.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/{tier}/IV?page=1&api_key={API_KEY}'
                time.sleep(1)
                response = requests.get(api_url)
                player_info = response.json()                    
                for player in player_info:
                    player_features.append([[player['wins'],
                                             player['losses'],
                                             player['veteran'],
                                             player['freshBlood'],
                                             player['hotStreak']]])
                    puuids.append({region.get(each): player['puuid']})
    return puuids, player_features

def get_champs(puuids, player_features, champ_map, regions):

    data_set = []
    champions = []
    for index, player in enumerate(puuids):
        print(f'Player number {index} being downloaded..')
        for region, puuid in player.items():
            mastery_url = f'https://{region}.api.riotgames.com/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}/top?count=15&api_key={API_KEY}'
            time.sleep(0.5)
            mastery_response = requests.get(mastery_url)
            if mastery_response.status_code != 200:
                print(mastery_response)
                time.sleep(15)
                continue
            champions_info = mastery_response.json()
            continent = get_continent(region, regions)
            playerid_url = f'https://{continent}.api.riotgames.com/riot/account/v1/accounts/by-puuid/{puuid}?api_key={API_KEY}'
            time.sleep(0.5)
            playerid_response = requests.get(playerid_url)
            if playerid_response.status_code != 200:
                print(playerid_response)
                time.sleep(15)
                continue
            playerid_info = playerid_response.json()
            if 'gameName' not in playerid_info:
                print('Skipping because player\'s name not exist.')
                continue
            for champion in champions_info:
                if str(champion['championId']) in champ_map:
                    if region not in champions and playerid_info['gameName'] not in champions:
                        champions.extend([playerid_info['gameName'] + "#" + playerid_info['tagLine'], region])
                if champion['championPoints'] >= 30000:
                    champions.extend([champ_map.get(str(champion['championId'])),champion['championPoints']])
                    champions.extend(player_features[index][0])
                    data_set.append(champions)
                    champions = []
    return data_set

def write_to_csv(data_set):

    with open('player_dataset_not_sparse.csv', 'a', newline='', encoding='utf-8') as csv_file:
        write = csv.writer(csv_file)
        write.writerows(data_set)
        csv_file.close()
    
def build_dataset():
    
    puuids, player_features = get_player_info(tiers, regions)
    champ_map = get_champ_map()
    data_set = get_champs(puuids, player_features, champ_map, regions)
    write_to_csv(data_set)
    
# build_dataset()

# def build_CBF_dataset():
#     with open('player_dataset_train.csv','r',encoding='utf-8') as csvfile:
#         datareader = csv.reader(csvfile)
#         for index,row in enumerate(datareader):
#             if index==0:
#                 with open('player_dataset_CBF_train.csv', 'a', newline='', encoding='utf-8') as csv_file:
#                     write = csv.writer(csv_file)
#                     write.writerow(row)
#                 with open('player_dataset_CBF_test.csv', 'a', newline='',encoding='utf-8') as csv_file:
#                     write = csv.writer(csv_file)
#                     write.writerow(row)
#             elif (index+1) % 3 != 0:
#                 with open('player_dataset_CBF_train.csv', 'a', newline='', encoding='utf-8') as csv_file:
#                     write = csv.writer(csv_file)
#                     write.writerow(row)
#                     csv_file.close()
#             else:
#                 with open('player_dataset_CBF_test.csv', 'a', newline='',encoding='utf-8') as csv_file:
#                     write = csv.writer(csv_file)
#                     write.writerow(row)
#                     csv_file.close()

# build_CBF_dataset()
