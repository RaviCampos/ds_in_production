import pickle
import inflection
import pandas as pd
import numpy as np
import math
import datetime

class Rossmann:
    def __init__(self):
        self.absolute_path = ""
        self.competition_distance_scaler = pickle.load( open( self.absolute_path + "parameter_scaler/competition_distance_scaler.pkl", "rb"))
        self.competition_time_in_months_scaler = pickle.load( open( self.absolute_path + "parameter_scaler/competition_time_in_months_scaler.pkl", "rb"))
        self.promo2_time_in_weeks_scaler = pickle.load( open( self.absolute_path + "parameter_scaler/promo2_time_in_weeks_scaler.pkl", "rb"))
        self.year_scaler = pickle.load( open( self.absolute_path + "parameter_scaler/year_scaler.pkl", "rb"))
        self.store_type_scaler = pickle.load( open( self.absolute_path + "parameter_scaler/store_type_scaler.pkl", "rb"))

        
    def data_cleaning(self, df1):

        # not going to use Sales or Custumer in any input, so no need to rename
        cols_old = ['Store', 'DayOfWeek', 'Date', 'Open', 'Promo',
               'StateHoliday', 'SchoolHoliday', 'StoreType', 'Assortment',
               'CompetitionDistance', 'CompetitionOpenSinceMonth',
               'CompetitionOpenSinceYear', 'Promo2', 'Promo2SinceWeek',
               'Promo2SinceYear', 'PromoInterval']
        snakecase = lambda x: inflection.underscore(x)

        # map returns an iterable, so it must be transformed into a list
        cols_new = list( map( snakecase, cols_old ) )

        # rename
        df1.columns = cols_new

        ## 1.3. Data Types
        df1["date"] = pd.to_datetime( df1["date"] )

        ## 1.5. Fill NA
        ### 1.5.1 competition_distance
        df1["competition_distance"] = df1["competition_distance"].apply( lambda x: 200000.0 if math.isnan(x) else x )

        ### 1.5.2 competition_open_since_month
        df1["competition_open_since_month"] = df1.apply( lambda x: x["date"].month if math.isnan(x["competition_open_since_month"]) else x["competition_open_since_month"], axis=1 )

        ### 1.5.3 competition_open_since_year
        df1["competition_open_since_year"] = df1.apply( lambda x: x["date"].year if math.isnan(x["competition_open_since_year"]) else x["competition_open_since_year"], axis=1 )

        ### 1.5.4 promo2_since_week
        df1["promo2_since_week"] = df1.apply( lambda x: x["date"].week if math.isnan(x["promo2_since_week"]) else x["promo2_since_week"], axis=1 )

        ### 1.5.5 promo2_since_year
        df1["promo2_since_year"] = df1.apply( lambda x: x["date"].year if math.isnan(x["promo2_since_year"]) else x["promo2_since_year"], axis=1 )

        ### 1.5.6 promo_interval
        df1["promo_interval"].fillna(0, inplace=True)

        month_map = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep", 8: "Oct", 9: "Nov", 10: "Dec", 11: "Nov", 12: "Dec"}

        df1["curr_month"] = df1["date"].dt.month.map( month_map )

        def in_promo_interval(x):
            if x["promo_interval"] == 0:
                return 0
            elif x["curr_month"] in x["promo_interval"].split(","):
                return 1
            else:
                return 0

        df1["is_curr_in_promo2"] = df1[["promo_interval", "curr_month"]].apply(in_promo_interval, axis=1)

        ## 1.6. Change types

        df1["competition_open_since_month"] = df1["competition_open_since_month"].astype( int )
        df1["competition_open_since_year"] = df1["competition_open_since_year"].astype( int )
        df1["promo2_since_week"] = df1["promo2_since_week"].astype( int )
        df1["promo2_since_year"] = df1["promo2_since_year"].astype( int )

        return df1
    
    
    def feature_engineering(self, df2):

        # YEAR
        df2["year"] = df2["date"].dt.year

        # MONTH
        df2["month"] = df2["date"].dt.month

        # DAY
        df2["day"] = df2["date"].dt.day

        # WEEK OF YEAR
        df2["week_of_year"] = df2["date"].dt.isocalendar().week

        # YEAR WEEK - just a date format (2015-30)
        df2["year_week"] = df2["date"].dt.strftime("%Y-%W")

        # COMPETITION SINCE
        df2["competition_since"] = df2.apply( lambda x: datetime.datetime( year=x["competition_open_since_year"], month=x["competition_open_since_month"], day=1 ), axis=1 )

        df2["competition_time_in_months"] = ( ( df2["date"] - df2["competition_since"] ) / 30 ).apply( lambda x: x.days ).astype(int)

        # PROMO2 SINCE
        df2["promo2_since"] = df2["promo2_since_year"].astype(str) + "-" + df2["promo2_since_week"].astype(str)

        df2["promo2_since"] = df2["promo2_since"].apply( lambda x: datetime.datetime.strptime( x + "-1", "%Y-%W-%w" ) - datetime.timedelta( days=7 ) )

        df2["promo2_time_in_weeks"] = ( ( df2["date"] - df2["promo2_since"] ) / 7 ).apply( lambda x: x.days ).astype(int)

        # ASSORTMENT
        df2["assortment"] = df2["assortment"].apply(lambda x: "basic" if x == "a" else "extra" if x == "b" else "extended")

        # STATE HOLIDAY
        df2["state_holiday"] = df2["state_holiday"].apply(lambda x: "public_holiday" if x == "a" else "easter_holiday" if x == "b" else "christmas" if x == "c" else "regular_day")

        # 3.0. Variable filtering
        ## 3.1. Row filtering
        df2 = df2[df2["open"] != 0]

        ## 3.2. Column filtering
        cols_drop = ["open", "promo_interval", "curr_month"]
        df2 = df2.drop( cols_drop, axis=1 )
        
        return df2
    
    
    def data_preparation(self, df5):
        
        cd_scaler = self.competition_distance_scaler 
        ctim_scaler = self.competition_time_in_months_scaler
        p2tiw_scaler = self.promo2_time_in_weeks_scaler
        y_scaler = self.year_scaler
        st_scaler = self.store_type_scaler

        # competition distance
        df5["competition_distance"] = cd_scaler.fit_transform( df5[["competition_distance"]].values )

        df5["competition_time_in_months"] = ctim_scaler.fit_transform( df5[["competition_time_in_months"]].values )

        # promo2_time_in_weeks
        df5["promo2_time_in_weeks"] = p2tiw_scaler.fit_transform( df5[["promo2_time_in_weeks"]].values )

        # year
        df5["year"] = y_scaler.fit_transform( df5[["year"]].values )

        ## 5.3. Transformation
        ### 5.3.1 Encoding
        # state_holiday - one hot encoding
        df5 = pd.get_dummies( df5, prefix=["state_holiday"], columns=["state_holiday"])

        # store_type - label encoding
        df5["store_type"] = st_scaler.fit_transform(df5["store_type"])

        # assortment - ordinal encoding
        assortment_dict = {"basic": 1, "extra": 2, "extended": 3}
        df5["assortment"] = df5["assortment"].map(assortment_dict)

        ### 5.3.2 Response Variable Transformation
        # df5["sales"] = np.log1p(df5["sales"])

        ### 5.3.3. Nature Transformation
        # day_of_week
        df5["day_of_week_sin"] = df5["day_of_week"].apply(lambda x: np.sin(x * (2 * np.pi / 7)))
        df5["day_of_week_cos"] = df5["day_of_week"].apply(lambda x: np.cos(x * (2 * np.pi / 7)))

        # month
        df5["month_sin"] = df5["month"].apply(lambda x: np.sin(x * (2 * np.pi / 12)))
        df5["month_cos"] = df5["month"].apply(lambda x: np.cos(x * (2 * np.pi / 12)))

        # day
        df5["day_sin"] = df5["day"].apply(lambda x: np.sin(x * (2 * np.pi / 30)))
        df5["day_cos"] = df5["day"].apply(lambda x: np.cos(x * (2 * np.pi / 30)))

        # week_of_year
        df5["week_of_year_sin"] = df5["week_of_year"].apply(lambda x: np.sin(x * (2 * np.pi / 52)))
        df5["week_of_year_cos"] = df5["week_of_year"].apply(lambda x: np.cos(x * (2 * np.pi / 52)))
        
        # manual col selection from boruta     
        cols_selected = ['store','promo','store_type','assortment','competition_distance','competition_open_since_month',
                    'competition_open_since_year','promo2','promo2_since_week','promo2_since_year',
                    'competition_time_in_months','promo2_time_in_weeks','day_of_week_sin','day_of_week_cos',
                    'month_sin','month_cos','day_sin','day_cos','week_of_year_sin','week_of_year_cos']

        return df5[cols_selected]
    
    def get_prediction( self, model, original_data, test_data ):
        # prediction
        prediction = model.predict( test_data )
        
        # join prediction into original data
        # must transform the prediction that is given in logm1 scale
        original_data["prediction"] = np.expm1(prediction)
        
        return original_data.to_json( orient="records", date_format="iso" ) 