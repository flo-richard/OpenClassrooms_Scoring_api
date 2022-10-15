import os
import pandas as pd
from pydantic import BaseModel
import dill as pickle
import lime
import numpy as np
import xgboost as xgb


class ScoringModel:
    def __init__(self):
        

        self.list_features = pickle.load(open('Pickles/features.pkl', 'rb'))
        self.all_features = self.list_features['all_features']
        self.continuous_features = self.list_features['continuous_features']
        self.categorical_features = self.list_features['categorical_features']        

        self.model = pickle.load(open('Pickles/xgb.pkl', 'rb')) # load trained model

        self.transformers = pickle.load(open('Pickles/transformers.pkl', 'rb')) #load transformers
        self.imputers = pickle.load(open('Pickles/imputers.pkl', 'rb')) #load imputers
        self.list_transformers = [i for i in self.transformers]  #list of transformers names (columns to be transformed)

        self.explainer = pickle.load(open('Pickles/explainer.pkl', 'rb')) #load lime explainer

        self.distributions = pickle.load(open('Pickles/distributions.pkl', 'rb'))



    def preprocessing(self, df: pd.DataFrame, req_info: dict):
        """Preprocess data (feature engineering, imputers, label encoders"""

        # Feature engineering

        df['INCOME_CREDIT_RATE'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['ANNUITY_INCOME_RATE'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        req_info['INCOME_CREDIT_RATE'] = df['INCOME_CREDIT_RATE'].values[0]
        req_info['ANNUITY_INCOME_RATE'] = df['ANNUITY_INCOME_RATE'].values[0]
        req_info['PAYMENT_RATE'] = df['PAYMENT_RATE'].values[0]

        # Imputers

        df[self.continuous_features] = self.imputers['imputer_mean'].transform(df[self.continuous_features])
        df[self.categorical_features] = self.imputers['imputer_most_frequent'].transform(df[self.categorical_features])

        # Label Encoders
        for i in self.list_transformers[:-1]:
            df[i] = self.transformers[i].transform(df[i])
        
        return df, req_info



    def predict(self, df: pd.DataFrame):

        prediction = self.model.predict(df.values)

        probas = [
            float(self.model.predict_proba(df.values.reshape(1,-1))[0][0]),
            float(self.model.predict_proba(df.values.reshape(1,-1))[0][1])
        ]

        return prediction, probas



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
        


# list = [
#     'Self-employed',
#     'School',
#     'University',
#     'Kindergarten',
#     'Government',
#     'Security Ministries',
#     'Legal Services',
#     'Postal',
#     'Military',
#     'Police',
#     'Security',
#     'Services',
#     'Religion',
#     'Medicine',
#     'Emergency',
#     'Electricity',
#     'Construction',
#     'Realtor',
#     'Housing',
#     'Hotel',
#     'Bank',
#     'Insurance',
#     'Mobile',
#     'Telecom',
#     'Culture',
#     'Advertising',
#     'Agriculture',
#     'Restaurant',
#     'Cleaning',
#     'Business Entity Type 1',
#     'Business Entity Type 2',
#     'Business Entity Type 3',
#     'Transport: type 1',
#     'Transport: type 2',
#     'Transport: type 3',
#     'Transport: type 4',
#     'Trade: type 1',
#     'Trade: type 2',
#     'Trade: type 3',
#     'Trade: type 4',
#     'Trade: type 5',
#     'Trade: type 6',
#     'Trade: type 7',
#     'Industry: type 1',
#     'Industry: type 2'
#     'Industry: type 3',
#     'Industry: type 4',
#     'Industry: type 5',
#     'Industry: type 6'
#     'Industry: type 7',
#     'Industry: type 8',
#     'Industry: type 9',
#     'Industry: type 10',
#     'Industry: type 11',
#     'Industry: type 12',
#     'Industry: type 13',
#     'Other'
# ]