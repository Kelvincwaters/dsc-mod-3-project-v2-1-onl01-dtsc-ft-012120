#### The project:
A credit card fraud detection kaggle dataset [here](https://www.kaggle.com/mlg-ulb/creditcardfraud). The dataset is without a dictionary and the majority of the features were obscured (maybe a result of a PCA dimensionality reduction to protect users identities and sensitive features). It expands 31 columns and over 280,000 rows @ 144MB and the majority of the features are continuous. There aren't any outliers and the data appears to be normalized. There's a major issue with this dataset, the extremely imbalanced target/dependent variable 'Class'. The columns and descriptions in the dataset are as follows:

```python
       df.columns
Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
       'Class'],
      dtype='object')
```

#### The models being considered: 
1. Logistic Regression
1. Decision Tree
1. Random Forest 
1. XGBoost

#### The metrics being considered:
1. Precison
1. Recall
1. Accuracy
1. F1-score

  > Note: The choice of which metric to pursue depends on the business objective and that objective being the precision metric, being able to ensure that the bank isn't "left holding the bag" on fraudulent transactions. Thus the focus of this notebook was on the precision metric.

#### Use of SMOTE
Oversampling the minority dependent variable was the chosen path and subsequently led to an almost a quater of a million additional data points. The initial imbalance for a 227451 to 394 split! 

```python
# instantiate a smote obj
smote = SMOTE(random_state= 42) 

# apply smote to training data NOT the testing data
X_train_smote, y_train_smote = smote.fit_sample(X_train, y_train)

# verifying the 'smoting' results!
# more data points is preferable than fewer data points when it comes to machine learning
print('Before SMOTE :', Counter(y_train))
print('After SMOTE :', Counter(y_train_smote))

Before SMOTE : Counter({0: 227451, 1: 394})
After SMOTE : Counter({0: 227451, 1: 227451})

```
```python
X_train_smote.shape
(454902, 30)

y_train_smote.shape
(454902,)

```
#### Use of pipelines (Houston we have a problem)
I attempted to utilized pipelines just for passing in GridSearchCV parameters since there was no use in creating dummies through sklearn via ColumnTransformer and a scaler seeing as how this dataset was already normalized. Using pipelines on the initial imbalanced data worked as expected but couldn't complete at all when presented with the smoted X_train_smoted, y_train_smoted parameters. An exhaustive google search returned an issue with SMOTE perhaps running on a single core (in a multicore system) but to no avail no workaround was discovered. 

```python
# logistic Regression
pipe_lr = Pipeline([('pca', PCA(n_components= 2)),
            ('clf', LogisticRegression(random_state= 42))])
# SVM
pipe_svm = Pipeline([('pca', PCA(n_components= 2)),
            ('clf', svm.SVC(random_state= 42))])
# Decision Tree            
pipe_dt = Pipeline([('pca', PCA(n_components= 2)),
            ('clf', tree.DecisionTreeClassifier(random_state= 42))])


# List of pipelines for ease of iteration
pipelines = [pipe_lr, pipe_svm, pipe_dt]

# Dictionary of pipelines and classifier types for ease of reference
pipe_dict = {0: 'Logistic Regression', 1: 'Support Vector Machine', 2: 'Decision Tree'}

# Fit the pipelines
for pipe in pipelines:
    pipe.fit(X_train_smote, y_train_smote) # will NOT complete with balanced smoted training data!
```

#### Use of GridSearchCV:
Each model was initially ran with default parameters which were recorded then subjected to a GridSearchCV which included the default parameters as well as additional ones, all with 5 K-Fold validation. 

```python
# default values are first in each dict list
param_grid = {
    'n_estimators': [10, 100],
    'criterion': ['gini', 'entropy'],
    'n_jobs': [None, -1]
}
gs_forest = GridSearchCV(forest, param_grid, cv=5) # 5 K fold
gs_forest.fit(X_train_smote, y_train_smote)
gs_forest.best_params_

{'criterion': 'entropy', 'n_estimators': 100, 'n_jobs': None} results! 
```

#### The Results:
1. First place Random Forest classifier @ 99% in both Precision and accucracy
1. Second place XGBoost @ 96% in Precision and 99% in accuracy
1. Third was Logitistic Regression (balanced) @ 93% accuracy and 99 accuracy
1. Last is a Decision Tree @ 89% precision and an accuracy of over 99% (as the others)

<img src= model_results.jpg width=300>


```python
# Instantiate and fit a RandomForestClassifier
forest = RandomForestClassifier(criterion= 'entropy', n_estimators= 100, n_jobs= None)

# fit
forest.fit(X_train_smote, y_train_smote)

# predict
forest_predict = forest.predict(X_test) # this can't be right, smote isn't applied to testing values

# train accuracy score
forest.score(X_train_smote, y_train_smote)

# test accuracy score
forest.score(X_test, y_test)      

# default 1.0

RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='entropy', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

Confusion Matrix:
 [[56863     1]
 [   21    77]]

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56864
           1       0.99      0.79      0.88        98

    accuracy                           1.00     56962
   macro avg       0.99      0.89      0.94     56962
weighted avg       1.00      1.00      1.00     56962
```

#### Future work:
Would be nice to know what the hidden features are and test for multicollinearity and maybe ascertain if there's any correlation of fraudulent transaction detection during a specific time of day, month ... this could clearly be indentified by the metrics captured with each and every credit card transaction. 