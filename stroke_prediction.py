
import numpy as np      # numpy and pandas use to perform data analysis
import pandas as pd     
import seaborn as sns   # Seaborn and matplotlib used to perform data visualization
import warnings         # Used to ignore warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,roc_curve)   # To check how accurate results are

# Loading the dataset using read_csv
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# head() is used to see first 5 rows of dataset
df.head(5)

# id column will not give any impact on data analysis so we can drop this column.
df.drop('id', axis=1, inplace=True)

# isnull() is used to check null values.
df.isnull().sum()

df['bmi'].fillna(df['bmi'].mode()[0], inplace=True)

df_num = df.select_dtypes(['int64','float64'])
df_cat = df.select_dtypes('object')

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

for col in df_num:
  mm = MinMaxScaler()
  df_num[col] = mm.fit_transform(df_num[[col]])

df_num.head()

# In our Dataset we have categorical data as well, so we need to convert this data
# into numerical format as ML understands only numbers.
# For this we will use Label Encoding.

for col in df_cat:
  le = LabelEncoder()
  df_cat[col] = le.fit_transform(df_cat[col])

df_cat.head()

# concat() is used to merge different data.
df_new = pd.concat([df_num, df_cat], axis = 1)
df_new.head()

df_new['stroke'].value_counts()

from sklearn.model_selection import train_test_split    # used to split data into training and testing part
from collections import Counter

x = df_new.drop('stroke', axis=1)
y = df['stroke']

df_new.shape

df_new['stroke'].value_counts()

df_new['stroke'] = df_new['stroke'].astype(int)

df_zero = df_new[df_new['stroke']==0]
df_one = df_new[df_new['stroke']==1]
print(len(df_zero),len(df_one))

((249+4861)/2) - 249

df_new_2 = pd.concat([df_one, df_zero.sample(n=2300)], axis=0)

from sklearn.ensemble import RandomForestClassifier

x = df_new_2.drop('stroke',axis=1)
y = df_new_2['stroke']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,random_state=1)
# Creating Random Forest Model
classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(x_train, y_train)

import pickle

filename = 'stroke-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

