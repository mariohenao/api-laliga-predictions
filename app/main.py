import uvicorn
#from typing import List
from pydantic import BaseModel
from fastapi import FastAPI
import tensorflow as tf
from use_model import FootballMachtPredictor
app = FastAPI()

model_instance = FootballMachtPredictor()

class Match(BaseModel):
    match: dict

@app.get('/')
def index():
    return {'message': 'API for predicting football match results from LaLiga.'}

@app.post('/predict')
def predict_review(data: Match):
    """
    FastAPI
    -----
    Args:
        data (Match): json file
    -------
    Returns:
        response: dict: {
            winner: predicted winner of the match, 
            confindence: confindence of the prediction
            }
    """
    data = data.dict()

    print('\nHERE')
    print(data['match'])
    print(type(data['match']))
    print('HERE\n')

    response = model_instance.get_response(data['match'])
    print(response)
    print('HERE\n')
    return {
        'winner': response['winner'],
        'confidence': str(response['confidence'])
        }

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)