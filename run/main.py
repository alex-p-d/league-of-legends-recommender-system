from blueprints.recommendation_website.recommendations import recommendation_bp
from flask import Flask

app = Flask(__name__, template_folder='templates')
app.register_blueprint(recommendation_bp)

if __name__ == '__main__':
      app.run(host='0.0.0.0',port=5000, debug=True)
