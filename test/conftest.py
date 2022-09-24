import pandas as pd
import pytest
from train_script.ml.data import process_data


@pytest.fixture()
def data_frame():
    data_path = './data/census_clean.csv'
    return pd.read_csv(data_path)


@pytest.fixture()
def data_frame_greater_than_50k():
    df_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native_country": "United-States"
    }
    return df_data


@pytest.fixture()
def data_frame_smarter_than_50k():
    df_data = {
        "age": 28,
        "workclass": "Private",
        "fnlgt": 338409,
        "education": "Bachelors",
        "education-num": 20,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native_country": "India"
    }
    return df_data


@pytest.fixture()
def processing_data_train_sample(data_frame):
    category_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X, y, encoder, lb = process_data(
        data_frame,
        categorical_features=category_features,
        label="salary",
        training=False,
        encoder=None,
        lb=None,
    )
    return X, y, encoder, lb
