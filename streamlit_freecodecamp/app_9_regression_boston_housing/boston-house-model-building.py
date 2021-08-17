# as homework, the model will be saved so there is no need to rebuild the model every time that input parameters are changed

import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
import pickle

boston = datasets.load_boston()

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = boston.copy() # a copy is sometimes needed so one can change one copy without changing the other.

# Separating X and y
X = pd.DataFrame(boston.data, columns=boston.feature_names) # independent / x variables (input features)
Y = pd.DataFrame(boston.target, columns=["MEDV"]) # target

# Build Regression Model
regr = RandomForestRegressor()
regr.fit(X, Y.values.ravel())

# Saving the model
pickle.dump(regr, open('boston_regr.pkl', 'wb'))
