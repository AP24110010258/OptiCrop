"""
OptiCrop - Flask Web Application
Smart Agricultural Production Optimization Engine

Routes:
  /              - Home page
  /about         - About page
  /findyourcrop  - Crop prediction form
  /predict       - POST endpoint for predictions
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
scaler_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler.pkl')

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Crop information for display
crop_info = {
    'rice': {'emoji': '🌾', 'season': 'Kharif (Rainy)', 'color': '#27ae60'},
    'maize': {'emoji': '🌽', 'season': 'Kharif / Rabi', 'color': '#f39c12'},
    'chickpea': {'emoji': '🫘', 'season': 'Rabi (Winter)', 'color': '#e67e22'},
    'kidneybeans': {'emoji': '🫘', 'season': 'Rabi (Winter)', 'color': '#c0392b'},
    'pigeonpeas': {'emoji': '🫛', 'season': 'Kharif (Rainy)', 'color': '#8e44ad'},
    'mothbeans': {'emoji': '🫘', 'season': 'Kharif (Summer)', 'color': '#d35400'},
    'mungbean': {'emoji': '🫛', 'season': 'Kharif / Summer', 'color': '#27ae60'},
    'blackgram': {'emoji': '🫘', 'season': 'Kharif (Rainy)', 'color': '#2c3e50'},
    'lentil': {'emoji': '🟤', 'season': 'Rabi (Winter)', 'color': '#a04000'},
    'pomegranate': {'emoji': '🍎', 'season': 'All Seasons', 'color': '#e74c3c'},
    'banana': {'emoji': '🍌', 'season': 'All Seasons', 'color': '#f1c40f'},
    'mango': {'emoji': '🥭', 'season': 'Summer', 'color': '#e67e22'},
    'grapes': {'emoji': '🍇', 'season': 'Winter / Spring', 'color': '#8e44ad'},
    'watermelon': {'emoji': '🍉', 'season': 'Summer', 'color': '#e74c3c'},
    'muskmelon': {'emoji': '🍈', 'season': 'Summer', 'color': '#f39c12'},
    'apple': {'emoji': '🍎', 'season': 'Winter / Spring', 'color': '#c0392b'},
    'orange': {'emoji': '🍊', 'season': 'Winter', 'color': '#e67e22'},
    'papaya': {'emoji': '🍈', 'season': 'All Seasons', 'color': '#f39c12'},
    'coconut': {'emoji': '🥥', 'season': 'All Seasons', 'color': '#795548'},
    'cotton': {'emoji': '☁️', 'season': 'Kharif (Rainy)', 'color': '#ecf0f1'},
    'jute': {'emoji': '🌿', 'season': 'Kharif (Rainy)', 'color': '#27ae60'},
    'coffee': {'emoji': '☕', 'season': 'Winter / Monsoon', 'color': '#6d4c41'},
}


@app.route('/')
def home():
    """Render the Home page."""
    return render_template('index.html')


@app.route('/about')
def about():
    """Render the About page."""
    return render_template('about.html')


@app.route('/findyourcrop')
def findyourcrop():
    """Render the FindYourCrop prediction form."""
    return render_template('findyourcrop.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Process the prediction form and return crop recommendation."""
    try:
        # Collect input values from the form
        nitrogen = float(request.form['nitrogen'])
        phosphorous = float(request.form['phosphorous'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create feature array and scale
        features = np.array([[nitrogen, phosphorous, potassium,
                              temperature, humidity, ph, rainfall]])
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)
        crop = prediction[0]

        # Get crop details
        info = crop_info.get(crop, {'emoji': '🌱', 'season': 'N/A', 'color': '#27ae60'})

        return render_template('findyourcrop.html',
                               prediction=crop.capitalize(),
                               emoji=info['emoji'],
                               season=info['season'],
                               color=info['color'],
                               nitrogen=nitrogen,
                               phosphorous=phosphorous,
                               potassium=potassium,
                               temperature=temperature,
                               humidity=humidity,
                               ph=ph,
                               rainfall=rainfall)

    except Exception as e:
        return render_template('findyourcrop.html', error=str(e))


if __name__ == '__main__':
    print("\n[OptiCrop] Server is running!")
    print("   Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True)
