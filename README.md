# FleetOpti ML

Ce dossier contient la partie Intelligence Artificielle et Machine Learning du projet **FleetOpti AI**.

## Structure du projet
- `data/` : Contient les datasets réels :
    - `vehicle_maintenance_data.csv` (Maintenance Prédictive)
    - `CO2 Emissions_Canada.csv` (Empreinte Carbone)
    - `logistics_dataset_with_maintenance_required.csv` (Données Logistiques)
- `notebooks/` : Analyses et explorations :
    - `01_Maintenance_Predictive.ipynb` : Analyse des pannes et maintenance.
    - `02_Analyse_Carbone.ipynb` : Facteurs d'émissions de CO2.
    - `03_Optimisation_Logistique.ipynb` : Efficacité opérationnelle et logistique.
- `models/` : Modèles exportés au format **ONNX**
- `src/` : Code source (prétraitement et entraînement)

## Objectifs
1. **Maintenance Prédictive** : Prédire les pannes à partir de l'historique de la flotte.
2. **Détection de Fraude** : Analyser les consommations de carburant et positions GPS.
3. **Empreinte Carbone** : Calculer et optimiser les émissions de CO2.

## Installation
```bash
pip install -r requirements.txt
```
