document.addEventListener('DOMContentLoaded', function() {
    // Éléments du DOM
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const predictionValue = document.getElementById('prediction-value');
    const loadingOverlay = document.getElementById('loading-overlay');
    const newPredictionBtn = document.getElementById('new-prediction-btn');
    const bmiCalculatorBtn = document.getElementById('bmi-calculator-btn');
    const bmiCalculator = document.getElementById('bmi-calculator');
    const calculateBmiBtn = document.getElementById('calculate-bmi');
    const bmiInput = document.getElementById('bmi');
    const weightInput = document.getElementById('weight');
    const heightInput = document.getElementById('height');
    const bmiResult = document.getElementById('bmi-result');
    const bmiValueElement = document.getElementById('bmi-value').querySelector('span');
    const bmiStatusElement = document.getElementById('bmi-status').querySelector('span');
    const bmiIndicator = document.getElementById('bmi-indicator');
    const stepInfo = document.getElementById('step-info');
    const stepResult = document.getElementById('step-result');
    const shareBtn = document.getElementById('share-btn');
    const globalTooltip = document.getElementById('global-tooltip');
    
    // Initialisation des tooltips
    initTooltips();
    
    // Afficher/masquer le calculateur d'IMC
    if (bmiCalculatorBtn) {
        bmiCalculatorBtn.addEventListener('click', function() {
            bmiCalculator.classList.toggle('hidden');
            // Faites défiler jusqu'au calculateur si affiché
            if (!bmiCalculator.classList.contains('hidden')) {
                bmiCalculator.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    }

    // Calculer l'IMC
    if (calculateBmiBtn) {
        calculateBmiBtn.addEventListener('click', function() {
            const weight = parseFloat(weightInput.value);
            const height = parseFloat(heightInput.value) / 100; // Convertir cm en m
            
            if (isNaN(weight) || isNaN(height) || weight <= 0 || height <= 0) {
                showNotification('Veuillez entrer un poids et une taille valides.', 'error');
                return;
            }
            
            const bmi = weight / (height * height);
            bmiInput.value = bmi.toFixed(1);
            
            // Déterminer la catégorie d'IMC et définir la position de l'indicateur
            let category = '';
            let position = '0%';
            
            if (bmi < 18.5) {
                category = 'Insuffisance pondérale';
                position = (bmi / 18.5) * 25 + '%';
            } else if (bmi < 25) {
                category = 'Corpulence normale';
                position = ((bmi - 18.5) / 6.5) * 25 + 25 + '%';
            } else if (bmi < 30) {
                category = 'Surpoids';
                position = ((bmi - 25) / 5) * 25 + 50 + '%';
            } else {
                category = 'Obésité';
                let maxBMI = 40;
                position = Math.min(((bmi - 30) / (maxBMI - 30)) * 25 + 75, 95) + '%';
            }
            
            // Mettre à jour les éléments d'IMC
            bmiValueElement.textContent = bmi.toFixed(1);
            bmiStatusElement.textContent = category;
            bmiIndicator.style.left = position;
            
            // Afficher le résultat de l'IMC
            bmiResult.classList.remove('hidden');
            
            // Notification
            showNotification(`Votre IMC est de ${bmi.toFixed(1)} (${category})`, 'success');
        });
    }

    // Revenir au formulaire
    if (newPredictionBtn) {
        newPredictionBtn.addEventListener('click', function() {
            resultContainer.classList.add('hidden');
            form.parentElement.classList.remove('hidden');
            
            // Mettre à jour les étapes
            stepResult.classList.remove('active');
            stepInfo.classList.add('active');
            
            // Réinitialiser le formulaire
            form.reset();
            
            // Masquer le calculateur d'IMC
            bmiCalculator.classList.add('hidden');
            bmiResult.classList.add('hidden');
        });
    }
    
    // Partage d'estimation
    if (shareBtn) {
        shareBtn.addEventListener('click', function() {
            const estimationValue = predictionValue.textContent;
            
            // Si l'API de partage est disponible
            if (navigator.share) {
                navigator.share({
                    title: 'Mon estimation de coût d\'assurance santé',
                    text: `Mon estimation de coût d'assurance santé est de ${estimationValue} par an.`,
                    url: window.location.href
                })
                .then(() => showNotification('Partagé avec succès!', 'success'))
                .catch(() => showNotification('Partage annulé', 'info'));
            } else {
                // Copier dans le presse-papier si le partage n'est pas disponible
                const text = `Mon estimation de coût d'assurance santé est de ${estimationValue} par an.`;
                navigator.clipboard.writeText(text)
                    .then(() => showNotification('Copié dans le presse-papier!', 'success'))
                    .catch(() => showNotification('Impossible de copier', 'error'));
            }
        });
    }

    // Soumission du formulaire
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault(); // Empêcher la soumission normale du formulaire
            
            // Vérifier si tous les champs sont remplis
            const requiredFields = form.querySelectorAll('[required]');
            for (let field of requiredFields) {
                if (!field.value) {
                    field.focus();
                    showNotification('Veuillez remplir tous les champs obligatoires', 'error');
                    return;
                }
            }
            
            // Afficher l'écran de chargement
            loadingOverlay.style.display = 'flex';
            
            // Récupérer les données du formulaire
            const formData = {
                age: parseInt(document.getElementById('age').value),
                sex: document.getElementById('sex').value,
                bmi: parseFloat(document.getElementById('bmi').value),
                children: parseInt(document.getElementById('children').value),
                smoker: document.getElementById('smoker').value,
                region: document.getElementById('region').value
            };
            
            // Valider les données
            if (formData.age < 18 || formData.age > 100) {
                showNotification("L'âge doit être entre 18 et 100 ans", 'error');
                loadingOverlay.style.display = 'none';
                return;
            }
            
            if (formData.bmi < 10 || formData.bmi > 50) {
                showNotification("L'IMC doit être entre 10 et 50", 'error');
                loadingOverlay.style.display = 'none';
                return;
            }
            
            if (formData.children < 0 || formData.children > 10) {
                showNotification("Le nombre d'enfants doit être entre 0 et 10", 'error');
                loadingOverlay.style.display = 'none';
                return;
            }
            
            // Envoyer les données à l'API
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(data => {
                        throw new Error(data.error || 'Une erreur est survenue');
                    });
                }
                return response.json();
            })
            .then(data => {
                // Afficher le résultat
                predictionValue.textContent = data.prediction;
                
                // Masquer le formulaire et afficher le résultat
                form.parentElement.classList.add('hidden');
                resultContainer.classList.remove('hidden');
                
                // Mettre à jour les étapes
                stepInfo.classList.remove('active');
                stepResult.classList.add('active');
                
                // Masquer l'écran de chargement
                loadingOverlay.style.display = 'none';
                
                // Faites défiler jusqu'au résultat
                resultContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            })
            .catch(error => {
                showNotification(error.message || 'Une erreur est survenue lors de la prédiction', 'error');
                loadingOverlay.style.display = 'none';
            });
        });
    }
    
    // Initialisation des tooltips
    function initTooltips() {
        // Ajouter des événements de survol pour les infobulles
        const tooltipTriggers = document.querySelectorAll('.tooltip');
        
        tooltipTriggers.forEach(tooltip => {
            tooltip.addEventListener('mouseover', function(e) {
                const text = this.querySelector('.tooltip-text').textContent;
                showTooltip(e, text);
            });
            
            tooltip.addEventListener('mouseout', function() {
                hideTooltip();
            });
            
            tooltip.addEventListener('mousemove', function(e) {
                moveTooltip(e);
            });
        });
    }
    
    // Fonctions de tooltip global
    function showTooltip(e, text) {
        globalTooltip.querySelector('#tooltip-content').textContent = text;
        globalTooltip.style.opacity = '1';
        moveTooltip(e);
    }
    
    function hideTooltip() {
        globalTooltip.style.opacity = '0';
    }
    
    function moveTooltip(e) {
        const tooltipWidth = globalTooltip.offsetWidth;
        const tooltipHeight = globalTooltip.offsetHeight;
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
        
        let x = e.clientX + scrollLeft + 10;
        let y = e.clientY + scrollTop + 10;
        
        // Éviter de sortir de l'écran
        if (x + tooltipWidth > window.innerWidth + scrollLeft) {
            x = e.clientX + scrollLeft - tooltipWidth - 10;
        }
        
        if (y + tooltipHeight > window.innerHeight + scrollTop) {
            y = e.clientY + scrollTop - tooltipHeight - 10;
        }
        
        globalTooltip.style.left = x + 'px';
        globalTooltip.style.top = y + 'px';
    }
    
    // Fonction pour afficher des notifications
    function showNotification(message, type = 'info') {
        // Supprimer les notifications existantes
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => {
            document.body.removeChild(notification);
        });
        
        // Créer la notification
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        // Choisir l'icône en fonction du type
        let icon = '';
        switch(type) {
            case 'success':
                icon = '<i class="fas fa-check-circle"></i>';
                break;
            case 'error':
                icon = '<i class="fas fa-exclamation-circle"></i>';
                break;
            case 'warning':
                icon = '<i class="fas fa-exclamation-triangle"></i>';
                break;
            default:
                icon = '<i class="fas fa-info-circle"></i>';
        }
        
        notification.innerHTML = `
            <div class="notification-icon">${icon}</div>
            <div class="notification-message">${message}</div>
            <button class="notification-close"><i class="fas fa-times"></i></button>
        `;
        
        document.body.appendChild(notification);
        
        // Ajouter la classe visible pour l'animation
        setTimeout(() => {
            notification.classList.add('visible');
        }, 10);
        
        // Ajouter un événement au bouton de fermeture
        notification.querySelector('.notification-close').addEventListener('click', function() {
            hideNotification(notification);
        });
        
        // Masquer automatiquement après 5 secondes
        setTimeout(() => {
            hideNotification(notification);
        }, 5000);
    }
    
    function hideNotification(notification) {
        notification.classList.remove('visible');
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }
}); 