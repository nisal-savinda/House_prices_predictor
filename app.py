from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

# Load model, scaler, and feature importance
model = pickle.load(open('house_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
feature_importance = pd.read_csv("feature_importance.csv")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get inputs
        area = request.form.get('area', type=float)
        bedrooms = request.form.get('bedrooms', type=int)
        bathrooms = request.form.get('bathrooms', type=int)
        stories = request.form.get('stories', type=int)

        # Validation
        if not (area and bedrooms and bathrooms and stories):
            return render_template('index.html', prediction_text="⚠️ Please fill in all fields.",
                                   area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories)
        if area <= 0 or bedrooms <= 0 or bathrooms <= 0 or stories <= 0:
            return render_template('index.html', prediction_text="⚠️ All values must be positive.",
                                   area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories)

        # Prepare input
        features = np.array([[area, bedrooms, bathrooms, stories]])
        features_scaled = scaler.transform(features)

        # Predict
        price = model.predict(features_scaled)[0]
        price_low = round(price * 0.9, 2)
        price_high = round(price * 1.1, 2)

        # Top features
        top_features = feature_importance.sort_values("importance", ascending=False).to_dict(orient='records')

        return render_template('index.html',
                               prediction_text=f"🏠 Estimated Price: ${price:,.2f} (Range: ${price_low:,.2f} - ${price_high:,.2f})",
                               area=area, bedrooms=bedrooms, bathrooms=bathrooms, stories=stories,
                               top_features=top_features)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
