# CustomerConversionPrediction
With the historical telephonic campaign details, prediction is done whether he will become customer to insurance company

The dataset has the features age , job , marital status, call type, education qualification , call duration, day , month, previous call outcome and number of calls 

and the target is to predict whether the contacted customer will opt an insurance - Customer conversion prediction

This is an Supervised Learning and binary classification problem.

The datset has 40K+ records and 35K+ records are under the category No. 
Since majority of the data falls under one target class, this is an imbalanced data. 

To balance the same, combination of both over sampling and under sampling techniques thru SMOTEEN (SMOTE + ENN-Edited Nearest neighbor ) is implemented.

Exploratory Data Analysis is done. The data is cleaned, encoded (Label encoding), Splitting , scaling (standardization) has been done.

Data visualization is done using plotly library 

The models are bulit using different Machine Learning Algorithms,

Logistic Regression
K-Nearest Neighbor Algorithm
Decision Tree
Max Voting Classifier
Random Forest Algorithm
XGBoost 

The final model XGBoost has AUROC 92 % and Accuracy 87 %. 

The model is deployed after pickling and web app is built using streamlit. 

