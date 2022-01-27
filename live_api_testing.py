from test_main import pos_examples, neg_examples
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default='http://localhost:8000/score')

args = parser.parse_args()
endpoint = args.endpoint

print(f'testing using endpoint: {endpoint}')


def test_example(example_data):
    response = requests.post(endpoint, json=example_data)
    msg = (f'api result: {response.status_code} '
           f'prediction: {response.json()} '
           f'expected: {example_data["salary"]}')
    print(msg)


for example in pos_examples + neg_examples:
    test_example(example)
