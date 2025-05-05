from flask import Blueprint, render_template, redirect, request, abort
from recommendationSystem.Recommendation import Recommender
from recommendationSystem.Player import Player
import os

recommendation_bp = Blueprint("recommendation",
                               __name__,
                               template_folder='templates',
                               static_folder='static',
                               static_url_path='/recommendation/static')

@recommendation_bp.route("/", methods=['GET'])
def home():
    return render_template('home.html')

@recommendation_bp.route("/recommend", methods=['POST'])
def recommend():
   
   id_tag = request.form.get('Gamename#Tag')
   region = request.form.get('Region')

   try:
        if len(id_tag) > 0 and len(region) > 0:
            recommendation_object = Recommender(id_tag.split('#')[0],id_tag.split('#')[1], region)
            player_object = Player(id_tag.split('#')[0],id_tag.split('#')[1], region)
            account_info = player_object.get_account_info()
            pfp = get_pfp(account_info.get('profileIconId'))
            recommended_champions = recommendation_object.recommend()
            champ_img = get_champ_img(recommended_champions)
            return render_template('recommendation_page.html',
                                    recommended_champions=recommended_champions,
                                    id_tag = id_tag,
                                    champ_img = champ_img,
                                    pfp = pfp,
                                    summoner_level = account_info.get('summonerLevel'))
        else:
            return 'failure'
   except ValueError:
       abort(404)
   

def get_champ_img(recommended_list):
    recommended_champs_img = []
    end_string = '_0.jpg'
    for filename in os.listdir('blueprints/recommendation_website/static/images'):
        for champ in recommended_list:
            if filename == champ[0]+end_string:
                recommended_champs_img.append(filename)
    recommended_champs_img = list(set(recommended_champs_img))              
    return recommended_champs_img

def get_pfp(id):
    end_string = '.png'
    for filename in os.listdir('blueprints/recommendation_website/static/profileicon'):
        if filename == str(id)+end_string:
            return filename
      
        
@recommendation_bp.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404