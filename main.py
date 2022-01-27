# Put the code for your API here.
import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

from fastapi import FastAPI, Query
from pydantic import BaseModel
from starter.ml.data import process_data
from starter.ml.model import inference

#from starter.train_model import cat_features

import pandas as pd
import pickle

model = pickle.load(open("model/model.pkl", 'rb'))
encoder = pickle.load(open("model/onehotencoder.pkl", 'rb'))
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

app = FastAPI()


class ScoringData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Query(None, alias='education-num')
    marital_status: str = Query(None, alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Query(None, alias='capital-gain')
    capital_loss: int = Query(None, alias='capital-loss')
    hours_per_week: int = Query(None, alias='hours-per-week')
    native_country: str = Query(None, alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 0,
                "native-country": "United-States",
            }
        }


@app.get("/")
async def greeting():
    return {"msg": "Welcome to the census predictor"}


@app.post("/score")
async def score(sd: ScoringData):
    df = pd.DataFrame(sd, columns=['0', '1'])\
        .set_index('0').T\
        .assign(salary='dummy')

    df.columns = [k.replace("_", "-") for k in df.columns]

    X_test, _, _, _ = process_data(
        df, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=None)

    pred = ">50K" if int(inference(model, X_test)) == 1 else "<=50K"

    return {'prediction': pred}
