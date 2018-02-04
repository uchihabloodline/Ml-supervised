import numpy as np
import pandas as pd

#The Machine learning alogorithm
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

# Test train split
from sklearn.cross_validation import train_test_split

# Just to switch off pandas warning
pd.options.mode.chained_assignment = None

# Used to write our model to a file
from sklearn.externals import joblib

#dirc = getcwd()
data = pd.read_csv("E:\MLprogs\\titanic_train.csv")

#print(data.head())
#print(data.columns)

median_age = data['age'].median()
print("Median age is {}".format(median_age))

data['age'].fillna(median_age, inplace = True)
data['age'].head()

data_inputs = data[["pclass", "age", "sex"]]
data_inputs.head()

expected_output = data[["survived"]]
expected_output.head()

data_inputs["pclass"].replace("3rd", 3, inplace = True)
data_inputs["pclass"].replace("2nd", 2, inplace = True)
data_inputs["pclass"].replace("1st", 1, inplace = True)
data_inputs.head()

data_inputs["sex"] = np.where(data_inputs["sex"] == "female", 0, 1)
data_inputs.head()

inputs_train, inputs_test, expected_output_train, expected_output_test   = train_test_split (data_inputs, expected_output, test_size = 0.33, random_state = 42)

print(inputs_train.head())
print(expected_output_train.head())

#reg = linear_model.LinearRegression()
rf = RandomForestClassifier (n_estimators=100)

rf.fit(inputs_train, expected_output_train)
#reg.fit(inputs_train, expected_output_train)
#train_prid = reg.predict(inputs_train)
#print(train_prid)

accuracy = rf.score(inputs_test, expected_output_test)
#accuracy = reg.score(inputs_test, expected_output_test)

print("Accuracy = {}%".format(accuracy * 100))

#joblib.dump(rf, "titanic_model1", compress=9)