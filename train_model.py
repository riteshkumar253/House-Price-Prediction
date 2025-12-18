import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Load data
housing = pd.read_csv("housing.csv")

# Create income category
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

# Train-test split
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    train_set = housing.loc[train_idx].drop("income_cat", axis=1)

housing = train_set.copy()

# Labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# Columns
num_attribs = housing.drop("ocean_proximity", axis=1).columns
cat_attribs = ["ocean_proximity"]

# Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

# Prepare data
housing_prepared = full_pipeline.fit_transform(housing)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(housing_prepared, housing_labels)

# Save files
import os
os.makedirs("model", exist_ok=True)

with open("model/house_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/pipeline.pkl", "wb") as f:
    pickle.dump(full_pipeline, f)

print("✅ Model and pipeline saved successfully")
