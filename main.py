import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
warnings.filterwarnings("ignore")


current_path = os.getcwd()
assets_path = os.path.join(current_path, "assets")
data = os.path.join(assets_path, "covid.csv")
pickle_path = os.path.join(assets_path, "covid_.pkl")
df = pd.read_csv(data)
df = df.dropna()

#converting numerical features to proper categories
df.sex.replace({1: 'Female', 2: 'Male'}, inplace=True)
df.pneumonia.replace({1: 'Yes', 2: 'No', 98:'Not Specified',99:'Not Specified', 97:'Not Specified'}, inplace=True)
df.pregnancy.replace({1: 'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.diabetes.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.copd.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.asthma.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.inmsupr.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.hypertension.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.other_disease.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.cardiovascular.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.obesity.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.renal_chronic.replace({1:'Yes', 2: 'No', 99:'Not Specified',98:'Not Specified', 97:'Not Specified'}, inplace=True)
df.contact_other_covid.replace({1: 'Yes', 2: 'No', 97:'Not Specified',99:'Not Specified',98:'Not Specified'}, inplace=True)
df.covid_res.replace({1: 1, 2: 0, 3:2}, inplace=True)
df.drop(['id','patient_type','entry_date','date_symptoms','date_died','intubed','tobacco','icu'],axis=1,inplace=True)
df=df[(df['covid_res']==0) | (df['covid_res']==1)]

x = df.drop("covid_res", axis=1,inplace=False)
y = df["covid_res"]
df.drop('covid_res',axis=1,inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
num_feat = df.select_dtypes(exclude='object').columns.tolist()
cat_feat = df.select_dtypes(include='object').columns.tolist()

col_transformer = ColumnTransformer(
    transformers=[
        ("ss", StandardScaler(), num_feat),
        ("ordinal", OrdinalEncoder(), cat_feat),
    ],
    remainder="drop",
)


model = GradientBoostingClassifier(
    learning_rate=0.005, 
    n_estimators=1200,
    max_depth=9, 
    min_samples_split=1200, 
    min_samples_leaf=60, 
    subsample=0.85, 
    random_state=10, 
    max_features=7,
    warm_start=True
)

pipe = Pipeline([("preprocessing", col_transformer), ("model", model)])
pipe.fit(x_train, y_train)
with open(pickle_path, "wb") as f:
    pickle.dump(pipe, f)
print("Pickle File created at {}".format(pickle_path))