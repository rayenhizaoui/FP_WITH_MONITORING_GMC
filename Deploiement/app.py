from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from predictor import predict_insurance_cost, get_prediction_stats
import flask_monitoringdashboard as dashboard
from prometheus_client import Counter, Histogram, generate_latest
import time
from dotenv import load_dotenv
from custom_dashboard import init_custom_dashboard

# Chargement des variables d'environnement
load_dotenv()

app = Flask(__name__)

# Configuration de l'application
app.config['VERSION'] = '1.0.0'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key_change_in_production')

# Configuration et initialisation du dashboard de monitoring
dashboard.config.init_from(file='monitoring_config.cfg')
dashboard.bind(app)

# Initialisation du dashboard personnalisé
app = init_custom_dashboard(app)

# Métriques Prometheus
PREDICTION_COUNT = Counter('model_predictions_total', 'Total number of predictions made')
PREDICTION_LATENCY = Histogram('model_prediction_latency_seconds', 'Time for model prediction')
PREDICTION_VALUE = Histogram('prediction_value', 'Distribution of prediction values', buckets=[1000, 5000, 10000, 15000, 20000, 30000, 50000])

# Charger le modèle et le scaler
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        
        # Récupérer les données du formulaire
        data = request.get_json()
        
        age = int(data['age'])
        sex = data['sex']
        bmi = float(data['bmi'])
        children = int(data['children'])
        smoker = data['smoker']
        region = data['region']
        
        # Vérifier les contraintes des données
        if age < 18 or age > 100:
            return jsonify({'error': 'L\'âge doit être entre 18 et 100 ans'}), 400
            
        if bmi < 10 or bmi > 50:
            return jsonify({'error': 'L\'IMC doit être entre 10 et 50'}), 400
            
        if children < 0 or children > 10:
            return jsonify({'error': 'Le nombre d\'enfants doit être entre 0 et 10'}), 400
        
        # Faire la prédiction
        prediction = predict_insurance_cost(age, sex, bmi, children, smoker, region)
        
        # Mettre à jour les métriques
        PREDICTION_COUNT.inc()
        PREDICTION_LATENCY.observe(time.time() - start_time)
        PREDICTION_VALUE.observe(prediction)
        
        # Formater la prédiction avec 2 décimales et le symbole $
        formatted_prediction = "${:,.2f}".format(prediction)
        
        return jsonify({
            'prediction': formatted_prediction,
            'raw_prediction': prediction
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': os.path.exists(MODEL_PATH),
        'scaler_loaded': os.path.exists(SCALER_PATH),
        'app_version': app.config['VERSION'],
        'timestamp': time.time()
    })

@app.route('/model-stats')
def model_stats():
    return jsonify(get_prediction_stats())

if __name__ == '__main__':
    app.run(debug=True)