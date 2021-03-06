import pickle
import pandas as pd
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
model = pickle.load( open( "/home/derfel/ht/ds/ds_in_production/model_rossmann.pkl", "rb" ) )

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
    
if __name__ == "__main__":
    app.run( "0.0.0.0" )