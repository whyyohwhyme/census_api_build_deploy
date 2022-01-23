# Script to train machine learning model.
import pickle 
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.
import pandas as pd 
data = pd.read_csv('../data/census_a_nowhitespace.csv')
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
preds = inference(model, X_test)
print(compute_model_metrics(preds, y_test))
with open('../model/model.pkl', 'wb') as f:
    pickle.dump(data, f)
print('saved model')

with open('../model/onehotencoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
print('saved onehotencoder')