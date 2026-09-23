import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv("homeprices.csv")

X = df[['area']]        # DataFrame with column name
y = df['price']

model = LinearRegression()

model.fit(X, y)

print("Weight (coef_):", model.coef_)
print("Bias (intercept_):", model.intercept_)

# Save model using Joblib
joblib.dump(model, "house_price_model.joblib")
print("Model saved successfully")

# Load model
loaded_model = joblib.load("house_price_model.joblib")

# Prediction
area_value = 5000
input_data = pd.DataFrame([[area_value]], columns=['area'])

prediction = loaded_model.predict(input_data)

print(f"Predicted price for {area_value} sqft:", prediction[0])