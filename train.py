import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score


# Load dataset
data = pd.read_csv('./data/hour.csv')

# Select only numerical columns (excluding the target 'cnt')
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
X = data[numerical_columns].drop(columns=['cnt'])
y = data['cnt']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict and calculate error for Linear Regression
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
print(f'Linear Regression MSE: {lr_mse}, R²: {lr_r2}')


# Train Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate error for Random Forest
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f'Random Forest MSE: {rf_mse}, R²: {rf_r2}')







