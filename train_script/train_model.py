# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import json
import os

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics


path_data = "../data/census_clean.csv"
# Add the necessary imports for the starter code.
print("[Infor] reading data from path {0}".format(path_data))

data = pd.read_csv(path_data, index_col=None)

# Add code to load in the data.

print("[Infor] Splitting data into training and testing set ")
# Optional enhancement, use K-fold
# cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
print("[Infor] Processing data  using category {0}".format(cat_features))


# Proces the test data with the process_data function.

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features,
    label="salary", training=False, encoder=encoder, lb=lb
)
print("[Infor] Trainging data")

# Train and save a model.
model = train_model(X_train, y_train)

print("[Infor] Doing inference")
# Do  prediction.
preds = inference(model=model, X=X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

score_saving_path = './performance_whole_data.json'
print("[Infor] Saving performace in the data file {0}"
      .format(score_saving_path))

with open(score_saving_path, "w") as f:
    json.dump(
        obj={"precision": precision, "recall": recall, "fbeta": fbeta},
        fp=f,
        indent=5,
    )
absolute_path = 'os.path.dirname(os.path.abspath(__file__))'
model_path = os.path.join(absolute_path, '../model/') + 'model.pkl'

print("[Infor] Saving the model in dir {0}".format(model_path))

# Save the model in `model_path`
joblib.dump(
    (
        model,
        encoder,
        lb,
        {'precision': precision, 'recall': recall, 'fbeta_scoree': fbeta}
    ),
    model_path
)
