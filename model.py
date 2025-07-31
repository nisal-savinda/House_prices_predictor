import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv("house_data.csv")

X = data[['area', 'bedrooms', 'bathrooms', 'stories']]
y = data['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print("✅ Model trained successfully")
print(f"📊 MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"📊 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"📊 R² Score: {r2_score(y_test, y_pred):.2f}")

# Save model and scaler
pickle.dump(model, open('house_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Save feature importances for frontend insights
importances = model.feature_importances_
importance_df = pd.DataFrame({"feature": X.columns, "importance": importances})
importance_df.to_csv("feature_importance.csv", index=False)

print("✅ Model saved successfully")