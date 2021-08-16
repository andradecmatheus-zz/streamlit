# in this project, it will be saved the model so there is no need to rebuild the model every time that input parameters are changed

import pandas as pd
penguins = pd.read_csv('penguins_cleaned.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy() # a copy is sometimes needed so one can change one copy without changing the other.
target = 'species' # select here what feature will be predicted
encode = ['sex','island']

# Convert categorical variable into dummy/indicator variables.
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) 
    df = pd.concat([df,dummy], axis=1)
    del df[col]
# now sex and island columns have their values transform in columns, where their values are 0 or 1;

# to perfom enconding in target variable
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)


# Separating X and y
X = df.drop('species', axis=1)
Y = df['species']

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
