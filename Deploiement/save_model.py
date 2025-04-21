# Exécuter ce script à partir de votre notebook pour sauvegarder le modèle et le scaler
import joblib
import os

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# Sauvegarder le modèle XGBoost optimisé
joblib.dump(best_xgb, 'models/xgboost_model.pkl')

# Sauvegarder le scaler
joblib.dump(scaler, 'models/scaler.pkl')

print("Modèle et scaler sauvegardés avec succès dans le dossier 'models/'")