import os
import pandas as pd
from pydantic import BaseModel
import dill as pickle
import lime
import numpy as np
import xgboost as xgb


class ScoringModel:
    def __init__(self):
        
        self.threshold = 0.696969
        self.list_features = pickle.load(open('Pickles/features.pkl', 'rb'))
        self.all_features = self.list_features['all_features']
        self.continuous_features = self.list_features['continuous_features']
        self.categorical_features = self.list_features['categorical_features']     

        self.data = pd.read_csv('Data/client_data_sample.csv', index_col=0)
        self.data = self.data.drop([i for i in self.data.columns if i not in self.all_features], axis=1)

        self.model = pickle.load(open('Pickles/xgb.pkl', 'rb')) # load trained model

        self.transformers = pickle.load(open('Pickles/transformers.pkl', 'rb')) #load transformers
        self.imputers = pickle.load(open('Pickles/imputers.pkl', 'rb')) #load imputers
        self.list_transformers = [i for i in self.transformers]  #list of transformers names (columns to be transformed)

        self.explainer = pickle.load(open('Pickles/explainer.pkl', 'rb')) #load lime explainer

        self.distributions = pickle.load(open('Pickles/distributions.pkl', 'rb'))

    def create_data_user(self, id : int):
        return pd.DataFrame(self.data.loc[[id]])

    def preprocessing(self, df: pd.DataFrame):
        """Preprocess data (feature engineering, imputers, label encoders"""

        # Feature engineering

        df['INCOME_CREDIT_RATE'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['ANNUITY_INCOME_RATE'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        df['DAYS_EMPLOYED'] = - df['DAYS_EMPLOYED'] 

        df_og = df.copy(deep=True)

        # Imputers

        df[self.continuous_features] = self.imputers['imputer_mean'].transform(df[self.continuous_features])
        df[self.categorical_features] = self.imputers['imputer_most_frequent'].transform(df[self.categorical_features])

        # Label Encoders
        for i in self.list_transformers:
            df[i] = self.transformers[i].transform(df[i])
        
        return df, df_og



    def predict(self, df: pd.DataFrame):

        proba = self.model.predict_proba(df.values.reshape(1,-1))[0][1]
        prediction = 1 if proba > self.threshold else 0

        return prediction, proba



    def explain_prediction(self, df: pd.DataFrame):

        expl_details = self.explainer.explain_instance(
            df.values.reshape(-1),
            self.model.predict_proba,
            num_features=6
        )



        return pd.DataFrame(expl_details.as_map()[1], columns=['Feature_idx', 'Scaled_value']), expl_details.as_list()

    def return_distributions(self, expl_details_map):

        names_main_features = []
        for i in expl_details_map['Feature_idx']:
            names_main_features.append(self.all_features[i])

        feat_to_plot = [i for i in self.distributions if i in names_main_features]

        distributions_to_plot = {}
        for i in feat_to_plot:
            distributions_to_plot[i] = self.distributions[i]
            
        return distributions_to_plot