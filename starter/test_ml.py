
from .ml.model import compute_model_metrics, train_model, inference
import pandas as pd
import pytest
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

@pytest.fixture()
def dummy_predictions_actual():

    actual = pd.Series([0,0,0,0,0,0,1,1,1,1,1,1])
    predic = pd.Series([0,0,0,1,1,1,0,0,1,1,1,1])

    return actual, predic

@pytest.fixture()
def dummy_model_data():
    X,y = make_classification() 
    return X,y

def test_model_metrics(dummy_predictions_actual):
    actual, predic = dummy_predictions_actual
    prec, recall, fbeta = compute_model_metrics(actual, predic)
    assert round(prec,2)==0.57, f"precision is {prec}, should be 0.5"
    assert round(recall,2)==0.67, f"recall is {recall}, should be 0.5"
    assert round(fbeta,2)==0.62, f"fbeta is {fbeta}, should be 0"

def test_model(dummy_model_data):
    X,y = dummy_model_data 

    train_X, test_X, train_y, test_y = train_test_split(X,y)
    model = train_model(train_X,train_y)
    preds = model.predict_proba(test_X)[:,1]
    auc = roc_auc_score(test_y, preds)
    
    assert auc > 0.6

def test_inference(dummy_model_data):
    X,y = dummy_model_data

    train_X, test_X, train_y, test_y = train_test_split(X,y)
    model = train_model(train_X,train_y)
    preds = inference(model, test_X)

    assert preds.max() <= 1
    assert preds.min() >= 0
