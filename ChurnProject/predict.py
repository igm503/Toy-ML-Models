import pickle
import torch
from flask import Flask
from flask import request

app = Flask('churn')

with open('mlp_model', 'rb') as file:   
    mlp = torch.load(file)

def predict(x):
    churn_prob = mlp.predict_proba(x)
    churn = churn_prob > .5
    return churn, churn_prob

@app.route('/predict', methods=['POST'])
def get_prediction():
    request