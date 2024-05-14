# import des packages
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
import re

### Préparation des données :

# Fonction qui charge des données :
def load_data(data_path):
    data = pd.read_csv(data_path)
    # On doit changer le nom des colonnes car LightLGBM ne supporte pas certains caractères :
    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in data.columns}
    new_n_list = list(new_names.values())
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
    data = data.rename(columns=new_names)
    data.set_index('SK_ID_CURR', inplace=True)
    if 'TARGET' in data.columns:
        data.drop('TARGET', axis = 1, inplace=True)
    return data

# Fonction qui charge le scaler et le modèle :
def load_scaler_and_model():
    with open("calibrated_lgbm_v2.pkl", 'rb') as f_in:
        model = pickle.load(f_in)
    with open("scaler_v2.pkl", 'rb') as f_in:
        scaler = pickle.load(f_in)
    return scaler, model

# Fonction qui scale les données :
def prepare_data(data, scaler):
    features = scaler.transform(data)
    features = pd.DataFrame(features, columns=data.columns, index=data.index)
    return features

# Fonction qui renvoie une liste des identifiants clients :
def get_clients_ids(features):
    clients_ids = features.index.to_list()
    return clients_ids

df = load_data("subset_test.csv")
scaler, model = load_scaler_and_model()
features = prepare_data(df, scaler)
clients_ids = get_clients_ids(df)
threshold = 0.377 # Déterminé lors de la modélisation

### Prédiction :

# Instantiate the flask object
app = Flask(__name__)

@app.route("/")
def welcome():
    return ("Welcome! Github Actions ok (: :) ?")

@app.route('/prediction/')
def print_id_list():
    return f'The list of valid client ids :\n\n{(clients_ids)}'

@app.route('/prediction/<int:customer_id>')
def prediction(customer_id):
    if customer_id in clients_ids:
        client_data = features.loc[customer_id].values.reshape(1, -1)
        proba = model.predict_proba(client_data)[0, 1]
        customer_info = {
            'id': customer_id,
            'proba_risk_class': proba.round(2),
            'class': 'no_risk' if proba <= threshold else 'risk' 
        }
        return jsonify(customer_info)
    else:
        return 'Customer_id is not valid.'
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)



