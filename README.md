# AYGAZ-MachineLearningBootcamp

TITANIC DATASET AND KAGGLE NOTEBOOK LINK:

https://www.kaggle.com/code/ozlemkaykc/aygaz-machinelearning-project




In this project, participants will work in two fundamental areas of machine learning: 

They will learn to classify data into categories for prediction purposes using supervised and unsupervised learning algorithms, predict continuous values based on input features, or cluster data based on these features.

The goal of this project is to provide students with practical experience in data analysis, model development, and evaluation techniques in the fields of Supervised and Unsupervised Learning within artificial intelligence and machine learning.




**ABOUT DATASET **

link: https://www.kaggle.com/datasets/rksensational/dataset


**Overview**

The data has been split into two groups:

*training set (train.csv)*

*test set (test.csv)*

The training set should be used to build your machine learning models. For the training set, we provide the outcome (also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, use the model you trained to predict whether or not they survived the sinking of the Titanic.

**Variable Notes**

* pclass: A proxy for socio-economic status (SES)
* 1st = Upper
* 2nd = Middle
* 3rd = Lower


* sibsp: The dataset defines family relations in this way.
* Sibling = brother, sister, stepbrother, stepsister
* Spouse = husband, wife 

* parch: The dataset defines family relations in this way.
* Parent = mother, father
* Child = daughter, son, stepdaughter, stepson
* Some children traveled only with a nanny, therefore parch=0 for them.
* 

  ![image](https://github.com/user-attachments/assets/1a93727d-b265-4591-9deb-abc987bb532e)



  **Data Preparation:**
​
Handled missing values by imputing mean values for Age and Fare, and mode values for Embarked and Cabin.
Created a new feature, FamilySize, based on SibSp and Parch.
Encoded categorical features (Sex and Embarked) using Label Encoding.


![image](https://github.com/user-attachments/assets/ed8ca70d-b3a4-4fe6-b54d-5d6a5bbfcc36)
-->
![image](https://github.com/user-attachments/assets/e40b1125-a0e0-4d9d-a56c-6ba6291e0bfb)


![image](https://github.com/user-attachments/assets/9f8668d8-b4bb-4e4b-819c-dde74f349a48)

![image](https://github.com/user-attachments/assets/f4ca677d-71fd-466f-a6d6-231096b580ab)


![image](https://github.com/user-attachments/assets/d476a31e-2c7d-41cf-87cb-2735781138ca)

​
**Model Training and Evaluation:**
​
Trained and evaluated several models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Linear Regression, K-Nearest Neighbors (KNN), and Support Vector Machine (SVM).
Compared model performances using accuracy scores.
Performed cross-validation and hyperparameter tuning for Logistic Regression using Grid Search and Randomized Search for Random Forest.

​
![image](https://github.com/user-attachments/assets/a12c6e38-4c9a-4b57-87ed-309ee6dd32ad)


![image](https://github.com/user-attachments/assets/fe86691a-72d9-443f-ba92-422ec9b4124e)



**Clustering Analysis:**
​
Applied K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models (GMM) for clustering.
Evaluated clustering performance using Silhouette Score.


![image](https://github.com/user-attachments/assets/44b2776c-ad87-4daf-b59b-b6fe2100ebb5)


​
**Enhanced Model Evaluation:**
​
Incorporated clustering labels into Logistic Regression and performed cross-validation.
Evaluated models with clustering features using Grid Search.

​
 ![image](https://github.com/user-attachments/assets/1cc2ffdd-8549-4786-b2f5-aad07dde63e7)

**Hyperparameter Tuning:**
​
Ensure you're using the best hyperparameters from your Grid Search or Randomized Search for each model in your final evaluations.

Feature Engineering:
​
Explore additional feature engineering techniques such as interaction terms or polynomial features to improve model performance.

​
![image](https://github.com/user-attachments/assets/9e195d44-5ac6-410d-9152-9a997cdb3fb4)

**Model Comparison:**
​
Extend the model comparison section by including additional metrics such as ROC AUC, precision-recall curves, and feature importance.

​
![image](https://github.com/user-attachments/assets/079f5aec-81a8-41e2-a8a0-fe0fce44e36c)


![image](https://github.com/user-attachments/assets/46f83c29-83d8-46c7-9cd3-ba032fdf4157)



**Visualization:**
​
Visualize feature importances from models like Random Forest or XGBoost.
Include more detailed plots for cross-validation performance and hyperparameter tuning results.


​![image](https://github.com/user-attachments/assets/7610fbad-fe5b-4d3f-9a37-c44cd6c6cd06)

​
**Automated Pipeline:**
​
Using an automated pipeline with Pipeline and ColumnTransformer from sklearn to streamline data preprocessing and model training.
​
