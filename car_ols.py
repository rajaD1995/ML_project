## Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
import pickle

## Step 2: Load the Dataset
original_df = pd.read_csv(r'C:\Users\USER\Documents\Python\Nareshit data analysis\stats and ML\ML\27th- l1, l2, scaling\lasso, ridge, elastic net\TASK-22_LASSO,RIDGE\car-mpg.csv')

df = original_df.copy()

## Step 4: Feature Selection and Preprocessing
# Drop irrelevant columns
df.drop(['car_name'], axis=1, inplace=True)

df = df.replace(r'[?!]', "", regex=True).replace(r'^\s*$', np.nan, regex=True)

df['hp'] = df['hp'].fillna(np.mean(pd.to_numeric(df['hp'])))

df['origin'] = df['origin'].replace({1: 'america', 2: 'europe', 3: 'asia'})
df = pd.get_dummies(df, columns=['origin'],dtype=int)

## Step 5: Define X and y
X = df.drop("mpg", axis=1)
y = df["mpg"]

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype('float64')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## Step 7: Feature Elimination based on p-values
# Iteratively remove features with p-value > 0.05
cols = list(X_train.columns)
pmax = 1
while len(cols) > 0:
    X_1 = sm.add_constant(X_train[cols])
    model = sm.OLS(y_train, X_1).fit()
    p_values = model.pvalues.iloc[1:]  # exclude intercept
    pmax = p_values.max()
    feature_with_p_max = p_values.idxmax()
    if pmax > 0.05:
        cols.remove(feature_with_p_max)
    else:
        break

selected_features = cols

## Step 8: Build Final Model using selected features
final_model = LinearRegression()
final_model.fit(X_train[selected_features], y_train)

# Predict
y_pred = final_model.predict(X_test[selected_features])

test_pred = pd.DataFrame({
    'Actual MPG': y_test.values,
    'Predicted MPG': y_pred
})

test_pred['Error'] = test_pred['Actual MPG'] - test_pred['Predicted MPG']
test_pred['Absolute Error'] = test_pred['Error'].abs()
print(test_pred.head())

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("Bias score:",final_model.score(X_train[selected_features], y_train))
print('Variance score:',final_model.score(X_test[selected_features], y_test))



## Step 9: Deployment-Ready Prediction Function
def predict_new(car_features: dict):
    """
    car_features = {
        'cyl': value,
        'disp': value,
        ...
    }
    """
    input_df = pd.DataFrame([car_features])
    return final_model.predict(input_df[selected_features])[0]


## Step 10: Example Future Prediction
example_car = {
    'cyl': 4,
    'disp': 140.0,
    'hp': 70,
    'wt': 2400,
    'acc': 19.5,
    'yr': 76,
    'car_type': 0,
    'origin_asia': 0,
    'origin_europe': 0,
    'origin_america': 1
}
print("Predicted MPG for new car:", predict_new(example_car))

predicted_mpg = predict_new(example_car)
tolerance = 1.0  # You can adjust this

# Filter original dataframe (before dropping car_name)
similar_cars = original_df[(original_df['mpg'] >= predicted_mpg - tolerance) &
                           (original_df['mpg'] <= predicted_mpg + tolerance)]

# Show only car names
print(similar_cars[['car_name', 'mpg']])



# Save the trained model to disk
import pickle
filename = 'ols_model.pkl'
with open(filename, 'wb') as file:
     pickle.dump(final_model, file)
print("Model has been pickled and saved as ols_model.pkl")

import os
print(os.getcwd())