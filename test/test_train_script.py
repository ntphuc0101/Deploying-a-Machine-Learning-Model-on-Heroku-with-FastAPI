from train_script.ml.data import process_data
# import train_script.ml.model as model
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
# from sklearn.ensemble import RandomForestClassifier
import os
import sys
import logging

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        '../train_script'))


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_process_data(data_frame):
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
    print(" check file ", data_frame.head(
        10))
    label = "salary"
    X_train, y_train, encoder, lb = process_data(data_frame.head(
        10), categorical_features=cat_features, label=label, training=True,
        encoder=None, lb=None)
    logger.info("Testing processing data")

    assert len(X_train) == len(y_train)
    assert isinstance(encoder, OneHotEncoder)
    assert isinstance(lb, LabelBinarizer)
    false_value = False
    assert data_frame.isnull().any().all() == false_value


# def test_train(processing_data_train_sample):
#     """
#     This function tests the capacity of the
#     train_model function to generate the  model.
#     """
#     X, y, encoder, lb = processing_data_train_sample
#
#     trained_model = model.train_model(X_train=X, y_train=y)
#
#     logger.info("Testing test train")
#
#     assert trained_model is not None
#     assert isinstance(trained_model, RandomForestClassifier)
#
#
# def test_model_metrics(processing_data_train_sample):
#     """
#     This function tests are to test the type of metrics.
#     """
#     logger.info("Test testing metrics")
#
#     X, y, encoder, lb = processing_data_train_sample
#     trained_model = model.train_model(X_train=X, y_train=y)
#
#     predictions = model.inference(trained_model, X)
#
#     precision, recall, fbeta = model.compute_model_metrics(y, predictions)
#     logger.info("Test testing metrics - check precision")
#     assert isinstance(precision, float)
#     logger.info("Test testing metrics - check recall")
#     assert isinstance(recall, float)
#     logger.info("Test testing metrics - check fbeta")
#     assert isinstance(fbeta, float)
