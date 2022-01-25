from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# df = pd.read_csv('data/census_a_nowhitespace.csv')
# df.query('salary == ">50K"').sample(1).to_dict(orient='record')
pos_examples = [{'age': 55, 'workclass': '?', 'fnlgt': 141807, 'education': 'HS-grad', 'education-num': 9, 'marital-status': 'Never-married', 'occupation': '?', 'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Male', 'capital-gain': 13550, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States', 'salary': '>50K'},
                {'age': 35, 'workclass': 'Private', 'fnlgt': 356238, 'education': 'Assoc-acdm', 'education-num': 12, 'marital-status': 'Never-married', 'occupation': 'Other-service', 'relationship': 'Not-in-family', 'race': 'White', 'sex': 'Female', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 80, 'native-country': 'United-States', 'salary': '>50K'}]
neg_examples = [{'age': 21, 'workclass': 'Private', 'fnlgt': 216181, 'education': 'Some-college', 'education-num': 10, 'marital-status': 'Never-married', 'occupation': 'Sales', 'relationship': 'Own-child', 'race': 'White', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 35, 'native-country': 'United-States', 'salary': '<=50K'},
                {'age': 42, 'workclass': 'Local-gov', 'fnlgt': 99554, 'education': 'Some-college', 'education-num': 10, 'marital-status': 'Married-civ-spouse', 'occupation': 'Transport-moving', 'relationship': 'Husband', 'race': 'White', 'sex': 'Male', 'capital-gain': 0, 'capital-loss': 0, 'hours-per-week': 40, 'native-country': 'United-States', 'salary': '<=50K'}]

def test_greeting():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"msg":"Welcome to the census predictor"}

def test_pos_outcome():
    for neg_example in neg_examples:
        example_refmt = {k.replace('-', '_'):v for k,v in neg_example.items()}
        response = client.post("/score", json=example_refmt)
        assert response.status_code == 200
        assert response.json()['prediction'] == example_refmt['salary']

def test_neg_outcome():
    for pos_example in pos_examples:
        example_refmt = {k.replace('-', '_'):v for k,v in pos_example.items()}
        response = client.post("/score", json=example_refmt)
        assert response.status_code == 200
        assert response.json()['prediction'] == example_refmt['salary']