# Model Card

Note that this model was done as a training exercise for the Udacity MLOPs Nanodegree 

## Model Details

This is a machine-learning driven approack to income estimation. The model aims to predict whether a persons salary is over 50k or under 50k USD using various demographic features. We make use of tree based methods to come up with a suitable model to predict the response. 

The high-level features are listed below 

- Age
- Working Status 
- Education 
- Occupation 
- Relationship status
- Gender
- Race
- Capital gains/losses
- Working hours per week 
- Country 

In addition, the data provides a weighting term (fnlwgt) that measures the population size, adjusting for various factors like ethnicity, age, rage and gender. 

## Intended Use

Income estimation has various use cases within business, policy and academia. This particular model is meant for research purposes and should not be used for commercial purposes as adjustments have not been made for business purposes. 

## Training Data
The data used for this exercise was obtained from the UCI Machine Learning Repository  [https://archive.ics.uci.edu/ml/datasets/census+income]

This is an extract of data from the 1994 Census Database.

## Evaluation Data
The base training data was split using a 80%/20% train and test split 

## Metrics

We use precision, recall and the f1 score to measure model performance. Test set performance is reported below 

|Precision | 0.633|
|Recall    | 0.615|
|F1 Score  | 0.621|

We also measured performance across various slices of the data. The table below shows the metrics when slicing across gender

|Gender | Precision| Recall| F1 Score|
|Male   |     0.638|  0.637|    0.637|
|Female |     0.609|  0.480|    0.537|

We see that the model is better at classifying males, there is a notable drop in recall for females. 

## Ethical Considerations
There have been historic income disparities among gender racial and ethnic groups. The data may be reflective of this and whatever models may trained will reinforce these inherent disparities. 

## Caveats and Recommendations
The source dataset is quite old, the industry and the world population has changed quite a bit since then so actual income characteristics would have changed. Updated census data can be sourced to see how the trends have changed 

No adjustments have been made to de-bias the model or the datasets

Basic data cleaning and modelling was done for this exercise. There should be more room to improve the model by using more advanced modelling techniques .

For this reason, we recommend that the model mainly be used for academic purposes
