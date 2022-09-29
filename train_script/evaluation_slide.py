"""
written by Nguyen Thai Vinh Phuc
This file is used to slide data on each feature and
do inference tasks to check performance of the model
"""
from collections import defaultdict
import pandas as pd
import joblib
# import json
from ml.data import process_data
from ml.model import inference, compute_model_metrics


def test_on_slice(model, encoder, lb, data, cat_feature, slice_value):
    """
    Parameters
    ----------
    model : trained model
    encoder: categorical features
    lb:
    data: cleaned data (panda frame)
    cat_feature: category feature
    slice_value: slide value

    Returns
    -------
    metrics of evaluation: precision, recall, fbeta (float)
    """
    sliced_data = data[data[cat_feature] == slice_value]

    X_test, y_test, encoder, lb = process_data(
        sliced_data, categorical_features=cat_features,
        label="salary", training=False, encoder=encoder, lb=lb)

    preds = inference(model=model, X=X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    return precision, recall, fbeta


if __name__ == '__main__':
    data_path = '../data/census_clean.csv'
    print("[info] reading the dataset {0}".format(data_path))
    data = pd.read_csv(data_path)
    model_path = '../model/model.pkl'
    print("[info] loading model - {0}".format(model_path))
    model, encoder, lb, metrics = joblib.load(model_path)

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
    data_slide_scores = defaultdict(list)
    data_slide_result = "evaluation_slice_data_new.txt"
    with open(data_slide_result, 'a') as f:
        for cat_feature in cat_features:
            for slice_value in data[cat_feature].unique():
                precision, recall, fbeta = test_on_slice(
                    model, encoder, lb, data, cat_feature, slice_value)
                result = f"""\n{'-' * 50}\nPerformance on sliced column -- \
                        {cat_feature} -- {slice_value}\n{'-' * 50} \
                        \nPrecision:\t{precision}\nRecall:
                        \t{recall}\nF-beta score:\t{fbeta}\n"""
                f.write(result)
                data_slide_scores[cat_feature].append(
                    {
                        "feature name": cat_feature,
                        "slice value": slice_value,
                        "precision": precision,
                        "recall": recall,
                        "fbeta": fbeta,
                    }
                )

    # data_slide_result = "evaluation_slice_data_new.json"
    # print(
    #     "[info] saving slide data evaluation of the model "
    #     "- {0}".format(data_slide_result))
    # with open(data_slide_result, "w") as f:
    #     json.dump(obj=data_slide_scores, fp=f, indent=5)
