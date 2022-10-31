import uvicorn
from fastapi import FastAPI, Request
from model import ScoringModel
import pandas as pd


app = FastAPI()
Model = ScoringModel()



@app.post('/getPrediction')
async def get_prediction(info : Request):

    id_number = await info.json()
    print(id_number)
    id = int(id_number['id'])
    print(id)
    if id not in Model.data.index:
        return {
            'Status': 'Error',
            'Message': 'Error: Unknown ID'
        }
    
    else:

        df = Model.create_data_user(id)

        df_og = df.copy(deep=True)
        print(df_og.to_dict())


        print('Preprocessing...')
        df = Model.preprocessing(df)
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
            'User info': df_og.to_dict(),
            'Explainer map': expl_details_map.to_dict('list'),
            'Explainer list': expl_details_list,
            'Distributions': distributions
        }

if __name__ == '__main__':
    main()