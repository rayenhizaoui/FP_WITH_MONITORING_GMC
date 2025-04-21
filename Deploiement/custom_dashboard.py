import flask
from flask import Blueprint, render_template, jsonify
import flask_monitoringdashboard as dashboard
from flask_monitoringdashboard.core.custom_graph import register_graph
from flask_monitoringdashboard.core.auth import secure
from flask_monitoringdashboard.database import session_scope
from flask_monitoringdashboard.database.endpoint import get_endpoints
from flask_monitoringdashboard.database.count import count_requests
from flask_monitoringdashboard.database.request import get_latencies_sample

import time
import datetime
import numpy as np
from predictor import get_prediction_stats
import random

custom_dashboard = Blueprint('custom_dashboard', __name__, url_prefix='/custom-dashboard')

# Configuration du style personnalisé
CUSTOM_STYLE = {
    'colors': {
        'primary': '#2c3e50',
        'secondary': '#3498db',
        'success': '#2ecc71',
        'danger': '#e74c3c',
        'warning': '#f39c12',
        'info': '#9b59b6'
    },
    'chart_colors': [
        'rgba(52, 152, 219, 0.8)',
        'rgba(46, 204, 113, 0.8)',
        'rgba(231, 76, 60, 0.8)',
        'rgba(155, 89, 182, 0.8)',
        'rgba(241, 196, 15, 0.8)'
    ]
}

@custom_dashboard.route('/')
@secure
def index():
    """Page d'accueil du tableau de bord personnalisé"""
    return render_template('custom_dashboard.html', style=CUSTOM_STYLE)

@custom_dashboard.route('/api/summary')
@secure
def api_summary():
    """Résumé des principales métriques pour l'API"""
    with session_scope() as db_session:
        endpoints = get_endpoints(db_session)
        now = datetime.datetime.utcnow()
        last_day = now - datetime.timedelta(days=1)
        
        total_requests = 0
        endpoint_data = []
        
        for endpoint in endpoints:
            count = count_requests(db_session, endpoint.id, last_day, now)
            total_requests += count
            endpoint_data.append({
                'name': endpoint.name,
                'count': count,
                'color': random.choice(CUSTOM_STYLE['chart_colors'])
            })
            
        # Top 5 des endpoints les plus utilisés
        endpoint_data.sort(key=lambda x: x['count'], reverse=True)
        top_endpoints = endpoint_data[:5]
        
    # Récupérer les statistiques du modèle
    model_stats = get_prediction_stats()
    
    return jsonify({
        'total_requests': total_requests,
        'top_endpoints': top_endpoints,
        'model_stats': model_stats
    })

@custom_dashboard.route('/api/performance')
@secure
def api_performance():
    """Métriques de performance pour l'API"""
    with session_scope() as db_session:
        endpoints = get_endpoints(db_session)
        
        performance_data = []
        for endpoint in endpoints:
            latencies = get_latencies_sample(db_session, endpoint.id, limit=100)
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                p95_latency = np.percentile(latencies, 95) if len(latencies) >= 20 else avg_latency
                performance_data.append({
                    'name': endpoint.name,
                    'avg_latency': round(avg_latency, 3),
                    'p95_latency': round(p95_latency, 3),
                    'samples': len(latencies)
                })
    
    return jsonify({
        'endpoints': performance_data
    })

# Enregistrer un graphique personnalisé pour le dashboard
def custom_model_statistics():
    """Graphique personnalisé pour les statistiques du modèle"""
    stats = get_prediction_stats()
    
    if stats['count'] > 0:
        # Simulation de données pour démonstration
        timestamps = []
        values = []
        execution_times = []
        
        # Créer des séries temporelles synthétiques
        now = time.time()
        day_in_seconds = 86400
        
        for i in range(30):
            timestamp = now - (30-i) * day_in_seconds // 30
            timestamps.append(datetime.datetime.fromtimestamp(timestamp))
            values.append(stats['avg_prediction'] * (0.9 + 0.2 * random.random()))
            execution_times.append(stats['avg_execution_time_ms'] * (0.8 + 0.4 * random.random()))
        
        # Formatage pour le graphique
        return {
            'data': {
                'x': timestamps,
                'prediction': values,
                'execution_time': execution_times
            }
        }
    else:
        return {
            'data': {
                'x': [],
                'prediction': [],
                'execution_time': []
            }
        }

# Fonction d'initialisation à appeler depuis app.py
def init_custom_dashboard(app):
    """Initialise le dashboard personnalisé"""
    app.register_blueprint(custom_dashboard)
    
    # Configuration du dashboard standard
    dashboard.config.group_by = lambda x: 'API' if 'api' in x else 'Pages'
    dashboard.config.version = lambda: app.config.get('VERSION', '1.0.0')
    
    return app 