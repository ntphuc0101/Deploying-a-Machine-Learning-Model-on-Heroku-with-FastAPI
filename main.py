# Put the code for your API here

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

from train_script.ml.data import process_data
from train_script.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull -f") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model_path = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    './model/model.pkl')

model, encoder, lb, metrics = joblib.load(model_path)

app = FastAPI()


class Data_Frame(BaseModel):
    """
    define Field  for Census prediction
    """
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Self-emp-not-inc")
    fnlgt: int = Field(..., example=215646)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=14, alias="education-num")
    marital_status: str = Field(...,
                                example="Married-civ-spouse",
                                alias="marital-status")
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="India", alias="native_country")


@app.get("/", summary="Root path API", description="Census prediction API")
async def root():
    return "[Info] This Get method for FastAPI inference."


@app.get("/send", summary="Test endpoint response",
         description="Should expect reception")
async def send():
    return {"send": "well received"}


@app.post(
    "/inference",
    summary="Predict API endpoint",
    description="predict classification result for Census data",
)
async def predict(data: Data_Frame):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    # model_path = os.path.join(
    #     os.path.dirname(
    #         os.path.abspath(__file__)),
    #     './model/model.pkl')
    #
    global model, encoder, lb, metrics
    data_dict = data.dict(by_alias=True)
    # convert data into a dictionary, then a pandas dataframe
    census_df = pd.DataFrame.from_dict([data_dict])

    X, y, encoder, lb = process_data(
        census_df,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False
    )
    y = inference(model, X)
    return {"output": lb.inverse_transform(y)[0]}
