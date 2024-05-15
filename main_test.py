from main import load_data, load_scaler_and_model, prepare_data, get_clients_ids
import pandas as pd
import pickle
import pytest
import sklearn.preprocessing
import lightgbm
import sklearn.calibration

# I want to create a test file that will be used with pytest to test the functions in the main.py file:

# Fonction qui teste si les données ne sont pas vides :
def not_empty_returns():
    data = load_data("subset_test.csv")
    scaler, model = load_scaler_and_model()
    features = prepare_data(data, scaler)
    clients_ids = get_clients_ids(data)
    assert type(scaler) == sklearn.preprocessing.StandardScaler
    assert type(model) == lightgbm.sklearn.LGBMClassifier
    assert data.shape != (0, 0)
    assert features.shape != (0, 0)
    assert len(clients_ids) != 0

# Fonction qui test que l'index de data est bien l'identifiant client :
def test_index_is_client_id():
    data = load_data("subset_test.csv")
    assert data.index.name == 'SK_ID_CURR'

