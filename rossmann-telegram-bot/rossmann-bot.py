import pandas as pd
import json
import requests

def load_data(store_id):
    # loading test dataset
    df10 = pd.read_csv( "/home/derfel/ht/ds/ds_in_production/data/test.csv" )
    df_store_raw = pd.read_csv( "/home/derfel/ht/ds/ds_in_production/data/store.csv" )

    # merge test dataset + store
    df_test = pd.merge( df10, df_store_raw, how="left", on="Store" )

    # choose a store for prediction
    df_test = df_test[df_test["Store"] == store_id]

    # remove closed days
    df_test = df_test[(df_test["Open"] != 0) & (~df_test["Open"].isnull())]
    df_test = df_test.drop("Id", axis=1)

    # convert to json
    data = json.dumps( df_test.to_dict( orient="records" ) )

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

# d2 = d1[["store", "prediction"]].groupby("store").sum().reset_index()

# for i in range(len(d2)):
#     store, prediction = d2.loc[i, "store"], d2.loc[i, "prediction"]
#     print(f"Store number {store} will sell {prediction} in the next 6 weeks")