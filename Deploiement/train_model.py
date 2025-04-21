import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

print("Chargement des données...")
# Charger les données d'assurance
data = pd.read_csv('data/insurance.csv')

# Traitement des variables catégorielles
data['sex'] = data['sex'].map({'female': 1, 'male': 0})
data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})

# One-hot encoding pour la région
region_dummies = pd.get_dummies(data['region'], prefix='region')
data = pd.concat([data, region_dummies], axis=1)
data.drop('region', axis=1, inplace=True)

# Feature engineering
print("Création des features...")
data['age_squared'] = data['age'] ** 2
data['bmi_squared'] = data['bmi'] ** 2

# Catégorisation de l'IMC
def bmi_category(bmi):
    if bmi < 18.5: return 0  # Sous-poids
    elif bmi < 25: return 1  # Normal
    elif bmi < 30: return 2  # Surpoids
    else: return 3  # Obésité

data['bmi_category'] = data['bmi'].apply(bmi_category)

# Interactions
data['smoker_bmi'] = data['smoker'] * data['bmi']
data['smoker_age'] = data['smoker'] * data['age']

# Séparation des features et de la target
X = data.drop('charges', axis=1)
y = data['charges']

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste des colonnes numériques pour le scaling
num_cols = ['age', 'bmi', 'children', 'age_squared', 'bmi_squared', 'smoker_bmi', 'smoker_age']

# Scaling
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Entraînement du modèle XGBoost
print("Entraînement du modèle...")
xgb_model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Évaluation du modèle
train_score = xgb_model.score(X_train, y_train)
test_score = xgb_model.score(X_test, y_test)
print(f"Score R² sur l'ensemble d'entraînement: {train_score:.4f}")
print(f"Score R² sur l'ensemble de test: {test_score:.4f}")

# Sauvegarde du modèle et du scaler
print("Sauvegarde du modèle et du scaler...")
joblib.dump(xgb_model, 'models/xgboost_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Modèle et scaler sauvegardés avec succès dans le dossier 'models/'") 