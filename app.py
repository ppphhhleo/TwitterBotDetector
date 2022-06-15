import numpy as np
from api.user_account_api import UserAccountAPI
from connectors.dnn_model_connector import  ModelConnector
from flask import Flask, render_template, request
app = Flask(__name__)
import tweepy

# You retrieve the user account and generate the input vector
# either by screen name or by the user id.

# Let's analyse a Bot account
CONSUMER_KEY: str = ""
CONSUMER_SECRET: str = ""
ACCESS_TOKEN: str = ""
ACCESS_TOKEN_SECRET: str = ""

def calculate(name):
    # 1. Set up the API object
    api: UserAccountAPI = UserAccountAPI(
        consumer_key=CONSUMER_KEY,
        consumer_secret=CONSUMER_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET)
    screen_name: str = name
    # print(iter)
    auth = tweepy.OAuthHandler(
        CONSUMER_KEY, CONSUMER_SECRET
    )
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    APII = tweepy.API(auth, proxy="http://127.0.0.1:1080")

    iter = APII.user_timeline(screen_name=screen_name, count=10)
    iters= {'tweet1':iter[0].text,'tweet2':iter[1].text,'tweet3':iter[2].text,'tweet4':iter[3].text}
    feature, vector = api.my_get_input_feature_vector_by_screen_name(
    screen_name=screen_name)

    credibility: float = api.get_user_account_credibility(
        input_user_embedding=vector)
    #user_id: str = id
    return feature,iters,credibility



def return_img_stream(img_local_path):
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/",methods=['POST'])
def get_identity():
    user_name= request.form['name']
    #user_id = request.form['id']  # search by user ID
    feature, tweets, credibility = calculate(user_name)
    print("return feature: ",feature)
    #print(feature[1])
    print("request:", request.form)
    # if('robot' in request.form):
    if credibility < 0.5:
        img_path = 'static/img/robot.jpg'
        img_stream = return_img_stream(img_path)
        return render_template('index.html',img_stream=img_stream, **feature, **tweets, credibility=credibility, judge="假")
        #return render_template('index.html', json=feature)
    else:
        img_path = 'static/img/human.jpg'
        img_stream = return_img_stream(img_path)
        return render_template('index.html',img_stream=img_stream,**feature,**tweets, credibility=credibility, judge="真人")
        #return render_template('index.html', json=feature)

if __name__ == "__main__":
    app.run()
