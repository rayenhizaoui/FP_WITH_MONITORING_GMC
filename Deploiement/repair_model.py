import joblib
import os
import xgboost as xgb
import numpy as np
import pandas as pd

print("Réparation du modèle XGBoost...")

# Chemins vers les fichiers
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')
OUTPUT_PATH = os.path.join('models', 'xgboost_model_fixed.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

try:
    # Charger le modèle problématique
    old_model = joblib.load(MODEL_PATH)
    print("Modèle chargé avec succès")
    
    # Convertir en modèle Booster (niveau inférieur)
    if hasattr(old_model, 'get_booster'):
        booster = old_model.get_booster()
        print("Booster extrait du modèle")
        
        # Sauvegarder dans un format intermédiaire
        temp_path = os.path.join('models', 'model.json')
        booster.save_model(temp_path)
        print(f"Modèle sauvegardé en format JSON: {temp_path}")
        
        # Recréer un nouveau modèle XGBoost
        new_model = xgb.XGBRegressor()
        new_model.load_model(temp_path)
        print("Nouveau modèle créé à partir du fichier JSON")
        
        # Sauvegarder le nouveau modèle
        joblib.dump(new_model, OUTPUT_PATH)
        print(f"Modèle réparé sauvegardé: {OUTPUT_PATH}")
        
        # Renommer les fichiers
        if os.path.exists(OUTPUT_PATH):
            backup_path = MODEL_PATH + '.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path) 
            os.rename(MODEL_PATH, backup_path)
            os.rename(OUTPUT_PATH, MODEL_PATH)
            print(f"Ancien modèle sauvegardé en: {backup_path}")
            print(f"Nouveau modèle prêt à l'emploi: {MODEL_PATH}")
    else:
        print("Le modèle n'a pas de méthode get_booster(), impossible de réparer")
        
except Exception as e:
    print(f"Erreur lors de la réparation du modèle: {str(e)}")
    raise 