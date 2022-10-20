# scoring_model_api

This non-interactive API uses a pre-saved trained Machine Learning (ML) model to compute a decision whether a bank client will be able to repay a credit. It takes in entry a .json file containing raw data and returns the prediction, the decision interpretability and various, relevant information.
Details on the ML model training can be found at https://github.com/flo-richard/OpenClassrooms_Projet7.

Being non-interactive, the API is to be used with a dashboard (details at https://github.com/flo-richard/OC_project7_dashboard), or with a notebook (see the api_request.ipynb notebook at https://github.com/flo-richard/OpenClassrooms_Projet7 for an example)

Both the API and the dashboard are deployed as heroku applications :

API : https://scoring-oc7.herokuapp.com/getPrediction

Dashboard : https://credit-score-dashboard-oc7.herokuapp.com/


KEYWORDS : API, ML model, decision interpretability
