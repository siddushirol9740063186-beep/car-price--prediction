import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# ---------------- LOAD DATA ----------------
data = pd.read_csv("car data.csv")

# ---------------- PREPROCESS ----------------
# Convert categorical to numeric
le = LabelEncoder()

data['Fuel_Type'] = le.fit_transform(data['Fuel_Type'])
data['Seller_Type'] = le.fit_transform(data['Seller_Type'])
data['Transmission'] = le.fit_transform(data['Transmission'])

# Feature Engineering
data['Car_Age'] = 2025 - data['Year']
data.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# ---------------- FEATURES & TARGET ----------------
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = RandomForestRegressor()
model.fit(X_train, y_train)

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")
