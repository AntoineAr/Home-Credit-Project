# import des packages
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, send_file
import re
import shap
import base64

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
def load_scaler_model_explainer():
    with open("calibrated_lgbm_v2.pkl", 'rb') as f_in:
        model = pickle.load(f_in)
    with open("scaler_v2.pkl", 'rb') as f_in:
        scaler = pickle.load(f_in)
    with open("shap_explainer_v2.pkl", 'rb') as f_in:
        explainer = pickle.load(f_in)
    return scaler, model, explainer

# Fonction qui scale les données :
def prepare_data(data, scaler):
    features = scaler.transform(data)
    features = pd.DataFrame(features, columns=data.columns, index=data.index)
    return features

# Fonction qui renvoie une liste des identifiants clients :
def get_clients_ids(features):
    clients_ids = features.index.to_list()
    return clients_ids

# Fonction qui crée le dataframe des données brutes :
def load_brut_data(path):
    df_brut = pd.read_csv(path)
    return df_brut

# Fonction qui renvoie les informations d'un client :
def get_client_infos(client_id, path):
    df_brut = load_brut_data(path)
    client = df_brut[df_brut['SK_ID_CURR'] == client_id].drop(['SK_ID_CURR'], axis=1)
    gender = client['CODE_GENDER'].values[0]
    age = client['DAYS_BIRTH'].values[0]
    age = int(np.abs(age) // 365)
    revenu = float(client['AMT_INCOME_TOTAL'].values[0])
    source_revenu = client['NAME_INCOME_TYPE'].values[0]
    montant_credit = float(client['AMT_CREDIT'].values[0])
    statut_famille = client['NAME_FAMILY_STATUS'].values[0]
    education = client['NAME_EDUCATION_TYPE'].values[0]
    ratio_revenu_credit = round((revenu / montant_credit) * 100, 2)
    dict_infos = {
        'sexe' : gender,
        'âge' : age,
        'revenu' : revenu,
        'source_revenu' : source_revenu,
        'montant_credit' : montant_credit,
        'ratio_revenu_credit' : ratio_revenu_credit,
        'statut_famille' : statut_famille,
        'education' : education
    }
    return dict_infos

df = load_data("data\subset_train.csv")
scaler, model, explainer = load_scaler_model_explainer()
features = prepare_data(df, scaler)
clients_ids = get_clients_ids(df)
threshold = 0.377 # Déterminé lors de la modélisation

# shap :
shap_values = explainer.shap_values(features)

# On ne retient que les explications pour la prédiction de la classe positive :
"""exp = shap.Explanation(shap_values[:, :, 1], 
                       shap_values.base_values[:,1], 
                       data = features.values,
                       feature_names = features.columns)"""

### Prédiction :

# Instantiate the flask object
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route("/")
def welcome():
    return ("Bienvenue sur l'API de prédiction de défaut")

@app.route('/prediction/')
def print_id_list():
    return f'The list of valid client ids :\n\n{(clients_ids)}'

@app.route('/prediction/<int:client_id>')
def prediction(client_id):
    if client_id in clients_ids:
        client_data = features.loc[client_id].values.reshape(1, -1)
        proba = model.predict_proba(client_data)[0, 1]

        client_infos = get_client_infos(client_id, "data\subset_train_brut.csv")

        customer_pred = {
            'id': client_id,
            'proba_risk_class': proba.round(2),
            'class': 'no_risk' if proba <= threshold else 'risk',
            'client_infos' : client_infos
        }

        return jsonify(customer_pred)
    else:
        return 'Client_id is not valid.'

# Fonction qui affiche la feature importance globale via un summary plot shap :
@app.get('/global_shap')
def global_shap():
    shap.summary_plot(shap_values[1], 
                      features = features.values,
                      feature_names = features.columns,
                      plot_type='violin',
                      max_display=15,
                      show=True)
    plt.savefig('global_shap.png')

# Fonction qui affiche la feature importance locale pour le client sélectionné :
@app.get('/local_shap/<int:client_id>')
def local_shap(client_id):
    if client_id in clients_ids:
        client_data = features.loc[client_id].values.reshape(1, -1)
        client_index = features.index.get_loc(client_id)
        exp = shap.Explanation(shap_values[1], 
                               explainer.expected_value[1], 
                               client_data, 
                               feature_names=features.columns)
        shap.plots.waterfall(exp[0])
        plt.savefig('local_shap.png')
    else:
        return 'Customer_id is not valid.'
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)