import pandas as pd
import numpy as np
import joblib
import os
import logging
import time
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_predictions.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('predictor')

# Chemins vers les fichiers de modèle
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Charger le modèle et le scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.info(f"Modèle et scaler chargés avec succès depuis {MODEL_PATH} et {SCALER_PATH}")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle ou du scaler: {str(e)}")
    raise

# Liste des colonnes numériques pour le scaling
num_cols = ['age', 'bmi', 'children', 'age_squared', 'bmi_squared', 'smoker_bmi', 'smoker_age']

# Logs des prédictions pour monitoring
prediction_log = []
MAX_LOG_SIZE = 1000  # Nombre maximum d'entrées à conserver

def bmi_category(bmi):
    """Catégoriser l'IMC selon les standards médicaux"""
    if bmi < 18.5: return 0  # Sous-poids
    elif bmi < 25: return 1  # Normal
    elif bmi < 30: return 2  # Surpoids
    else: return 3  # Obésité

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Prédit le coût d'assurance pour un nouveau client.
    
    Paramètres:
    - age: âge du client (int)
    - sex: sexe du client ('male' ou 'female')
    - bmi: indice de masse corporelle (float)
    - children: nombre d'enfants couverts (int)
    - smoker: si le client est fumeur ('yes' ou 'no')
    - region: région du client ('northeast', 'northwest', 'southeast', 'southwest')
    
    Retourne:
    - Le coût d'assurance prédit
    """
    start_time = time.time()
    
    # Log des inputs pour suivi
    logger.info(f"Demande de prédiction: age={age}, sex={sex}, bmi={bmi}, children={children}, smoker={smoker}, region={region}")
    
    # Convertir en valeurs numériques
    sex_val = 1 if sex == 'female' else 0
    smoker_val = 1 if smoker == 'yes' else 0
    
    # Créer un dictionnaire pour les régions
    regions = {
        'northeast': [1, 0, 0, 0],
        'northwest': [0, 1, 0, 0],
        'southeast': [0, 0, 1, 0],
        'southwest': [0, 0, 0, 1]
    }
    region_vals = regions.get(region, [0, 0, 0, 0])  # Default en cas de région inconnue
    
    # Créer un DataFrame pour le nouveau client
    new_data = pd.DataFrame({
        'age': [age],
        'sex': [sex_val],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker_val],
        'region_northeast': [region_vals[0]],
        'region_northwest': [region_vals[1]],
        'region_southeast': [region_vals[2]],
        'region_southwest': [region_vals[3]]
    })
    
    # Ajouter les features engineered
    new_data['age_squared'] = new_data['age'] ** 2
    new_data['bmi_squared'] = new_data['bmi'] ** 2
    new_data['bmi_category'] = new_data['bmi'].apply(bmi_category)
    new_data['smoker_bmi'] = new_data['smoker'] * new_data['bmi']
    new_data['smoker_age'] = new_data['smoker'] * new_data['age']
    
    try:
        # Scaling
        new_data[num_cols] = scaler.transform(new_data[num_cols])
        
        # Prédire
        prediction = model.predict(new_data)[0]
        
        # Convertir le résultat numpy.float32 en float Python standard
        prediction_float = float(prediction)
        
        # Calculer le temps d'exécution
        execution_time = time.time() - start_time
        
        # Enregistrer les détails de la prédiction pour le monitoring
        prediction_entry = {
            'timestamp': datetime.now().isoformat(),
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'prediction': prediction_float,
            'execution_time_ms': execution_time * 1000
        }
        prediction_log.append(prediction_entry)
        
        # Limiter la taille du log
        if len(prediction_log) > MAX_LOG_SIZE:
            prediction_log.pop(0)
        
        logger.info(f"Prédiction réussie: {prediction_float:.2f}$ (temps: {execution_time*1000:.2f}ms)")
        return prediction_float
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}")
        raise

def get_prediction_stats():
    """Récupère des statistiques sur les prédictions récentes"""
    if not prediction_log:
        return {
            'count': 0,
            'avg_prediction': 0,
            'avg_execution_time_ms': 0,
            'min_prediction': 0,
            'max_prediction': 0
        }
    
    predictions = [entry['prediction'] for entry in prediction_log]
    execution_times = [entry['execution_time_ms'] for entry in prediction_log]
    
    return {
        'count': len(prediction_log),
        'avg_prediction': sum(predictions) / len(predictions),
        'avg_execution_time_ms': sum(execution_times) / len(execution_times),
        'min_prediction': min(predictions),
        'max_prediction': max(predictions),
        'recent_predictions': prediction_log[-10:] # 10 dernières prédictions
    }