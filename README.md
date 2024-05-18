## Réalisation d'un modèle de scoring visant à établir la probabilité de défaut de paiement de clients.

*Projet réalisé dans le cadre de la formation Datascience d'Openclassrooms*

### Contexte :
L’entreprise souhaite mettre en œuvre un outil de “scoring crédit” pour calculer la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s’appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.)

*Données disponibles ici : https://www.kaggle.com/c/home-credit-default-risk/data*

### Objectifs :

   - Réaliser une analyse exploratoire des données en se basant sur des kernels existants : https://github.com/AntoineAr/Home-Credit-Project/blob/main/EDA/EDA.ipynb
   - Elaborer un modèle de scoring qui puisse prédire automatiquement la probabilité de défaut d'un client et son statut "à risque" ou "non risqué" : https://github.com/AntoineAr/Home-Credit-Project/blob/main/Mod%C3%A9lisation/Mod%C3%A9lisation.ipynb
        - *il fallait prendre en compte, dans la démarche de modélisation, un déséquilibre des classes à prédire ainsi qu'un contexte métier impliqaunt des coûts plus élevés pour certaines mauvaises prédictions.*
        - un tracking des performances des modèles (et la recherche des meilleurs hyperparamètres) a été effectué via MLFlow
   - Analyser les features importance globale et locales (nous avons utilisé SHAP)
   - Créer une API (voir https://github.com/AntoineAr/Home-Credit-Project/blob/main/main.py) permettant d'interagir avec le modèle et, à partir d'une requête sur un identifiant client, d'obtenir : 
       - Sa probabilité de défaut;
       - son statut;
       - quelques informations le concernant.
   - Déployer cette API sur le cloud (heroku : https://application-credit-7ba79bc598e5.herokuapp.com/)
   - Réaliser des tests unitaires, nous avons utiliser pytest pour cela (fichier https://github.com/AntoineAr/Home-Credit-Project/blob/main/main_test.py)
   - Effectuer une analyse de datadrift à l'aide de la librairie evidently : https://github.com/AntoineAr/Home-Credit-Project/blob/main/Data_drift/Data_drift_analysis.ipynb (les fichiers application_train.csv et application_test.csv sont disponibles au lien kaggle ci-dessus).