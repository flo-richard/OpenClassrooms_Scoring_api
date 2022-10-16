import uvicorn
from fastapi import FastAPI, Request
from model import ScoringModel
import pandas as pd


app = FastAPI()
Model = ScoringModel()



@app.post('/getPrediction')
async def get_prediction(info : Request):
    req_info = await info.json()
    df = pd.DataFrame([req_info])



    print('Preprocessing...')
    df, req_info = Model.preprocessing(df, req_info)
    print('Preprocessing ok')

    print('Computing prediction...')
    prediction, probas = Model.predict(df)        
    print('Prediction :', prediction)

    print('Computing explainer...')
    expl_details_map, expl_details_list = Model.explain_prediction(df)
    print('Done')

    distributions = Model.return_distributions(expl_details_map)
    print('Done')

    return {
        'Status': 'Success',
        'Prediction': int(prediction),
        'Prediction probabilities': probas,
        'User info': req_info,
        'Explainer map': expl_details_map.to_dict('list'),
        'Explainer list': expl_details_list,
        'Distributions': distributions
    }

if __name__ == '__main__':
    main()