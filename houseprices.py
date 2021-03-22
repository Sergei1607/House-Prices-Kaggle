# -*- coding: utf-8 -*-

# =============================================================================
# =============================================================================
# =============================================================================
# # # House Prices -- Competition Kaggle
# =============================================================================
# =============================================================================
# =============================================================================

# =============================================================================
# Goal : Predict house prices.
# =============================================================================

# =============================================================================
# Methodology :  Superviced learning and regression techniques.
# =============================================================================

# =============================================================================
# =============================================================================
#                                   Code
# =============================================================================
# =============================================================================

# =============================================================================
# Import packages
# =============================================================================

# General packages 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Machine learning packages

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

# Preprocessing packages

from sklearn.preprocessing import scale

# =============================================================================
# =============================================================================
# # Global configurations
# =============================================================================
# =============================================================================

pd.set_option("display.max_columns", 12)
plt.style.use("ggplot")


# =============================================================================
# Importing data 
# =============================================================================

houseprices = pd.read_csv("train.csv")
housepricesid = pd.read_csv("train.csv")

houseprices.columns

# =============================================================================
# Detecting missing values
# =============================================================================

houseprices.info()

houseprices.isnull().sum()

# we have missing values 

# =============================================================================
# Due to the large amount of features, we need to find witch ones are
# the most important.
# =============================================================================

corr = houseprices.corr()

f, ax = plt.subplots(figsize=(20, 20))

sns.heatmap(corr, vmax=.8, square=True);

# =============================================================================
# There is to much information and we should only focus in the higuer
# correlations.
# =============================================================================

important_corr = houseprices.corr()

# this way we can extract the features that have strong correlations
# with our target feature (SalePrice)

features_with_high_corr = important_corr.index[abs(important_corr["SalePrice"])>0.3]

# we do the plot

f, ax = plt.subplots(figsize=(20, 12))
sns.heatmap(houseprices[features_with_high_corr].corr(),annot=True,cmap="RdYlGn")

# =============================================================================
# Extracting the features with high correlation
# =============================================================================

houseprices = houseprices[features_with_high_corr]

houseprices

# =============================================================================
# Finding missing values
# =============================================================================

houseprices.isnull().sum()

houseprices.shape

# we have missing values but they are not that much to consider 
# removing the feature so we need to deal with them 

# =============================================================================
# Dealing with missing values of LotFrontage
# =============================================================================

houseprices["LotFrontage"].unique()
sns.boxplot(houseprices["LotFrontage"])

#LofFrontage has a lot of unique values and outliers so we will fill
# the NAs with the median.

houseprices["LotFrontage"].unique()
sns.boxplot(houseprices["LotFrontage"])

houseprices["LotFrontage"].fillna(houseprices["LotFrontage"].median(), 
                                  inplace=True)

# =============================================================================
# Dealing with missing values of GarageYrBlt
# =============================================================================

houseprices["GarageYrBlt"].unique()
sns.boxplot(houseprices["GarageYrBlt"])

#LofFrontage has a lot of unique values but not outliers so we will fill
# the NAs with the mean.

houseprices["GarageYrBlt"].fillna(houseprices["GarageYrBlt"].mean(), 
                                  inplace=True)

# Now we dont have and missing values in our features.

# =============================================================================
# Dealing with missing values of "MasVnrArea"
# =============================================================================

sns.boxplot(houseprices["MasVnrArea"])

houseprices["MasVnrArea"].isnull().sum()

houseprices["MasVnrArea"].fillna(houseprices["MasVnrArea"].median(), 
                                  inplace=True)

# Now we dont have and missing values in our features.

# =============================================================================
# Adding the id column 
# =============================================================================

houseprices["Id"] = housepricesid["Id"]

# =============================================================================
# Now that we have all data clean lets make a simple regression model
# keeping in mind that we can improve a lot here
# =============================================================================

# First lets generate our X and y data

X = houseprices.drop("SalePrice", axis =1)

y = houseprices["SalePrice"]


# Now we need to split our data in train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=30)

# Create the model

lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred= lr.predict(X_test)

y_pred

lr.score(X_test, y_test)

# Doing some cross validation

cv_results = cross_val_score(lr, X, y, cv=10)

























# =============================================================================
# =============================================================================
# # Test data 
# =============================================================================
# =============================================================================

test= pd.read_csv("test.csv")
testid = pd.read_csv("test.csv")


ft= features_with_high_corr.drop('SalePrice')

test  = test[ft]

test.isnull().sum()

# =============================================================================
# Dealing with missing values of LotFrontage
# =============================================================================


#LofFrontage has a lot of unique values and outliers so we will fill
# the NAs with the median.


test["LotFrontage"].fillna(test["LotFrontage"].median(), 
                                  inplace=True)

# =============================================================================
# Dealing with missing values of GarageYrBlt
# =============================================================================


#LofFrontage has a lot of unique values but not outliers so we will fill
# the NAs with the mean.

test["GarageYrBlt"].fillna(test["GarageYrBlt"].mean(), 
                                  inplace=True)


# =============================================================================
# Dealing with missing values of "MasVnrArea"
# =============================================================================


test["MasVnrArea"].fillna(test["MasVnrArea"].median(), 
                                  inplace=True)


# =============================================================================
# Dealing with missing values of "TotalBsmtSF, GarageArea, BsmtFinSF1 "
# =============================================================================

columns = ["TotalBsmtSF", "GarageArea", "BsmtFinSF1"]
for i in columns:
    test[i].fillna(test[i].median(), inplace= True)

# =============================================================================
# Dealing with missing values of "GarageCars "
# =============================================================================

test["GarageCars"].unique()
test["GarageCars"].fillna(test["GarageCars"].value_counts().index[0], 
                                  inplace=True)

test.isnull().sum()

# now we dont have missing values.

test["Id"] = testid["Id"]




# =============================================================================
# # Creating the submition with logistic regression and test data
# =============================================================================

lr = LinearRegression()

lr.fit(X, y)

submisionlr1 = pd.DataFrame({"Id": test["Id"],
                            "SalePrice":lr.predict(test)})


submisionlr1.to_csv("submisionlr1.csv", index=False)

y_predict = np.floor(np.expm1(lr.predict(test)))

y_predict

# =============================================================================
# # Creating the submition with ridge regression
# =============================================================================

ridge = Ridge(alpha = 0.1, normalize = True)

ridge.fit(X, y)


submisionridge3 = pd.DataFrame({"Id": test["Id"],
                            "SalePrice":ridge.predict(test)})


submisionridge3.to_csv("submisionridge3.csv", index=False)


# =============================================================================
# # Creating the submition with lasso regression
# =============================================================================

lasso = Lasso(alpha = 0.1, normalize = True)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

submisionlasso1 = pd.DataFrame({"Id": test["Id"],
                            "SalePrice":lasso.predict(test)})


submisionlasso1.to_csv("submisionlasso1.csv", index=False)

# lasso doesnt work

# =============================================================================
# # hyperparameter tunning 
# =============================================================================

lrParams = {"alpha":[0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,]}



gridSearch = GridSearchCV(ridge, param_grid=lrParams, cv=10, n_jobs = -1)
    
gridSearch.fit(X, y)
    
bestParams, bestScore = gridSearch.best_params_,(gridSearch.best_score_*100, 2)

bestParams  
bestScore  

# random forest

rfr = RandomForestRegressor()

rfr.fit(X, y)

submisionrf1 = pd.DataFrame({"Id": test["Id"],
                            "SalePrice":rfr.predict(test)})


submisionrf1.to_csv("submisionrf1.csv", index=False)



rfParams = {"criterion":["mse","mae"],
             "n_estimators":[10, 15, 20, 25, 30],
             "min_samples_leaf":[1, 2, 3],
             "min_samples_split":np.arange(3,8), 
             "max_features":["sqrt", "auto", "log2"],
             "random_state":[44]}

gridSearch = GridSearchCV(rfr, param_grid=rfParams, cv=10, n_jobs = -1)
    
gridSearch.fit(X, y)

bestParams, bestScore = gridSearch.best_params_,(gridSearch.best_score_*100, 2)
bestParams  

### otr0 randomforest

rfr = RandomForestRegressor(criterion= 'mse', max_features= 'auto', 
                            min_samples_leaf= 1, min_samples_split= 5, 
                            n_estimators= 25, random_state= 44)

rfr.fit(X, y)

submisionrf2 = pd.DataFrame({"Id": test["Id"],
                            "SalePrice":rfr.predict(test)})


submisionrf2.to_csv("submisionrf2.csv", index=False)
