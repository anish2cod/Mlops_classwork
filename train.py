import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Tracking")
# Load dataset
data = pd.read_csv('hour.csv')

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

def lr_mlflow():
    with mlflow.start_run(run_name="Linear Regression"):

        # Log model, parameters, and metrics
        mlflow.log_param("Model Type", "Linear Regression")
        mlflow.log_metric("MSE", lr_mse)
        mlflow.log_metric("R²", lr_r2)
        
        mlflow.sklearn.log_model(lr_model,"linear_regression_model")
        print("linear regression model is logged in mlflow")
lr_mlflow()

# Train Random Forest Regressor Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and calculate error for Random Forest
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f'Random Forest MSE: {rf_mse}, R²: {rf_r2}')





# Train and log Random Forest model
def rf_mlflow():
    with mlflow.start_run(run_name="Random Forest"):

        # Log model, parameters, and metrics
        mlflow.log_param("Model Type", "Random Forest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("MSE", rf_mse)
        mlflow.log_metric("R²", rf_r2)
        
        mlflow.sklearn.log_model(rf_model,"random_forest_model")

rf_mlflow()

print('Training complete. Models saved, and experiments logged with MLflow.')



mlflow.set_experiment("Best Model")
if rf_r2 > lr_r2 :
    rf_mlflow()
else:
    lr_mlflow()
