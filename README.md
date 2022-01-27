# Census data model and api 

This is a submission for the Udacity MlOps Nanodegree. This repo contains code to build a model, store it in DVC and then expose the model using FastAPI 

You can link heroku to this repo and deploy the app to a heroku dyno 

## Model card

More information on the model can be obtained by looking at the model card (named model_card.md)

## Model training

The code to train the model resides in the starter directory. The entrypoint is train_model.py. You can execute the following to retrain the model 

```sh 
   cd ./starter
   python train_model.py
```
You might need to pull the data first using dvc (if you have access to the s3 bucket - raise an issue and I can help you with this)

```sh 
   dvc pull 
```

We have also included unit tests for the machine learning components. Run them using the following shell commands

```sh 
   cd ./starter
   pytest -v .
```

## Running the API locally

The main.py file contains the REST API implementation. You can run the server using the following 

```sh
   uvicorn main:app
````

Once that is done, you can use your browser and navigate to the /docs endpoint (usually https://localhost:8000/docs) to get the details on the model api and run a few test cases using the swagger UI 

You can also run the live api testing script 

```sh
   python live_api_testing.py --endpoint http://localhost:8000/score
```
We have also included unit tests for the API 

```sh
   pytest -v .
```
