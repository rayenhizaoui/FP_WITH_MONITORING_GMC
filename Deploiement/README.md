# Prédicteur de Coûts d'Assurance Santé

Application web basée sur Flask et XGBoost pour prédire les coûts d'assurance santé en fonction des caractéristiques des clients.

## Fonctionnalités

- Formulaire interactif pour saisir les informations du client
- Calcul d'IMC intégré
- Prédiction du coût d'assurance en temps réel
- Interface utilisateur responsive et moderne
- Système de monitoring complet

## Architecture technique

- **Backend** : Flask (Python)
- **Modèle ML** : XGBoost
- **Frontend** : HTML/CSS/JavaScript
- **Données** : Jeu de données CSV (insurance.csv)
- **Monitoring** : Flask-MonitoringDashboard, Prometheus

## Installation

1. Cloner le dépôt
2. Installer les dépendances :
   ```
   pip install -r requirements.txt
   ```
3. Créer un fichier `.env` basé sur le modèle `env.example` :
   ```
   cp env.example .env
   ```
4. Entraîner le modèle (nécessaire lors de la première installation) :
   ```
   python train_model.py
   ```
5. Lancer l'application :
   ```
   python app.py
   ```
6. Accéder à l'application via http://localhost:5000

## Entraînement du modèle

Le modèle est entraîné à partir du fichier CSV situé dans `data/insurance.csv`. Le script `train_model.py` :

- Prétraite les données
- Crée des fonctionnalités avancées
- Entraîne un modèle XGBoost
- Sauvegarde le modèle et le scaler dans le dossier `models/`

## Structure du projet

- `app.py` : Application Flask principale
- `predictor.py` : Fonctions pour effectuer les prédictions
- `train_model.py` : Script pour entraîner et sauvegarder le modèle
- `data/` : Dossier contenant les données d'entraînement
- `models/` : Dossier contenant les modèles sauvegardés
- `templates/` : Templates HTML pour l'interface utilisateur
- `static/` : Fichiers statiques (CSS, JavaScript, images)
- `monitoring_config.cfg` : Configuration du tableau de bord de monitoring

## Monitoring

Le projet inclut plusieurs outils de monitoring pour suivre les performances de l'application et du modèle :

### Flask-MonitoringDashboard
- Interface web accessible à l'adresse `/dashboard`
- Suivi des temps de réponse des endpoints
- Détection des outliers
- Analyse des performances par version d'application
- Authentification requise (configurée dans `monitoring_config.cfg`)

### Prometheus Metrics
- Métriques exposées à l'endpoint `/metrics`
- Comptage des prédictions
- Histogramme des temps de latence
- Distribution des valeurs prédites

### Endpoints de monitoring

- `/health` : Vérification de l'état de l'application et du modèle
- `/model-stats` : Statistiques sur les prédictions récentes
  - Nombre total de prédictions
  - Valeurs moyenne, minimale et maximale
  - Temps d'exécution moyen
  - Dernières prédictions effectuées

### Logs
- Journalisation des prédictions dans `model_predictions.log`
- Format structuré pour faciliter l'analyse

## Déploiement

Ce projet peut être déployé sur n'importe quelle plateforme supportant Python et Flask :

1. Pour le déploiement en production, utilisez :
   ```
   gunicorn app:app
   ```
2. Configurez les variables d'environnement appropriées pour la production
3. Pour un monitoring avancé, intégrez avec Grafana ou une solution de monitoring cloud

## Auteur

Rayen HIZAOUI


