import os
import pickle
import pandas as pd
import requests
import json

from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# ===========================
# === TO START THE SERVER ===
# run python handler.py
# make a request to the open port via terminal executed file
# or just a exexuted cell in jupyter notebook
# (there is already a test cell in this project's jupyter notebook)
# ===========================

# loading model
model = pickle.load( open( "model/model_rossmann.pkl", "rb" ) )



# ==================================

#    functions for telegram bot 

# ==================================

# constants
TOKEN = "2044464556:AAHRDANSa-7h-0EPPmW3sY4uoEgWq7cw98s"
# https://api.telegram.org/bot2044464556:AAHRDANSa-7h-0EPPmW3sY4uoEgWq7cw98s/getMe
# https://api.telegram.org/bot2044464556:AAHRDANSa-7h-0EPPmW3sY4uoEgWq7cw98s/sendMessage?chat_id

def send_telegram_message(chat_id,text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}"

    r = requests.post(url, json={"text": text})

    print(f"Status Code {r.status_code}")

    return None

def load_dataset(store_id):
    # loading test dataset
    df10 = pd.read_csv( "/home/derfel/ht/ds/ds_in_production/data/test.csv" )
    df_store_raw = pd.read_csv( "/home/derfel/ht/ds/ds_in_production/data/store.csv" )

    # merge test dataset + store
    df_test = pd.merge( df10, df_store_raw, how="left", on="Store" )

    # choose a store for prediction
    df_test = df_test[df_test["Store"] == store_id]

    if not df_test.empty:
        # remove closed days
        df_test = df_test[(df_test["Open"] != 0) & (~df_test["Open"].isnull())]
        df_test = df_test.drop("Id", axis=1)

        # convert to json
        data = json.dumps( df_test.to_dict( orient="records" ) )
    else:
        data = "error"

    return data

def predict(store_data):
    
    # API call

    # for local enviroment
    # url = "http://0.0.0.0:5000/rossmann/predict"

    # for heroku enviroment
    url = "https://powerful-rossmann-model.herokuapp.com/rossmann/predict"

    header = {"Content-type": "application/json"}
    data = store_data

    r = requests.post( url, data=data, headers=header )
    print( f"Status Code: {r.status_code}" ) 

    d1 = pd.DataFrame( r.json(), columns=r.json()[0].keys() )

    return d1

def parse_message(message):
    chat_id = message["message"]["chat"]["id"]
    store_id = message["text"]

    try:
        store_id = int(store_id)
    except ValueError:
        store_id = "error"
    
    return chat_id, store_id

# initialize API
app = Flask( __name__ )

@app.route( "/rossmann/predict", methods=["POST"] )
def rossmann_predict():

    test_json = request.get_json()
    print(test_json)
    
    if test_json: # there is data in the request
        
        if isinstance( test_json, dict ):
            test_raw = pd.DataFrame( test_json, index=[0] )
            
        else:
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys() )
            
        # instantiate Rossmann class
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning( test_raw )
        
        # feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        # data preparation
        df3 = pipeline.data_preparation( df2 )
        
        # prediction
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response
        
    else:
        
        return Response( "{}", status=200, mimetype="application/json" )


@app.route( "/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        message = request.get_json()

        chat_id, store_id = parse_message( message )

        if store_id != "error":
            # load data
            data = load_dataset(store_id)

            if data != "error":
                # prediction
                d1 = predict(data)

                # group by
                d2 = d1[["store", "prediction"]].groupby("store").sum().reset_index()

                # send message
                store, prediction = d2["store"].values[0], d2["prediction"].values[0]
                msg = f"Store number {store} will sell {prediction} in the next 6 weeks"
                
                send_telegram_message(chat_id, msg)
                return Response( "Ok", status=200 )

            else:
                send_telegram_message(chat_id, "Store id could not be found in the dataset")
                return Response( "Ok", status=200 )
        
        else:
            send_telegram_message(chat_id, "You must send a number")
            return Response( "Ok", status=200 )
    else:
        return "<h1>Rossmann Telegram API</h1>"
        
    
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run( host="0.0.0.0", port=port )