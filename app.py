import uvicorn
from fastapi import FastAPI, Request
from model import ScoringModel
import pandas as pd


app = FastAPI()
Model = ScoringModel()

# @app.get('/')
# def index():
#     return {'message': 'Hello World'}

#@app.get('/{name}')
#def get_name(name: str):    
#    return {'message': f'Hello, {name}'}

# if __name__ == '__main__':
#     uvicorn.run(app, hist='127.0.0.1', port=8000)


@app.post('/getPrediction')
async def get_prediction(info : Request):
    req_info = await info.json()
    df = pd.DataFrame([req_info])

    #print(req_info)

    # Ajouter les fonctions qui font le traitement ici
    print('Preprocessing...')
    df_scaled, req_info = Model.preprocessing(df, req_info)
    print('Preprocessing ok')

    print('Computing prediction...')
    prediction = Model.predict(df_scaled)        
    print('Prediction :', prediction)

    print('Computing explainer...')
    expl_details_map, expl_details_list = Model.explain_prediction(df)
    print('Done')

    distributions = Model.return_distributions(expl_details_map)
    

    return {
        'Status': 'Success',
        'Prediction': int(prediction),
        'User info': req_info,
        'Explainer map': expl_details_map.to_dict('list'),
        'Explainer list': expl_details_list,
        'Distributions': distributions
    }

if __name__ == '__main__':
    main()