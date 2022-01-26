from test_main import pos_examples, neg_examples
import requests
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--endpoint", default='http://localhost:8000/score')

args = parser.parse_args()
endpoint = args.endpoint

print(f'testing using endpoint: {endpoint}')


def test_example(example_data):
    example_refmt = {k.replace('-', '_'): v for k, v in example_data.items()}
    response = requests.post(endpoint, json=example_refmt)
    msg = (f'api result: {response.status_code} '
           f'prediction: {response.json()} '
           f'expected: {example_refmt["salary"]}')
    print(msg)


for example in pos_examples + neg_examples:
    test_example(example)
