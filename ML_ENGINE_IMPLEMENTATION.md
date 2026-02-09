# 🧠 FleetOpti AI - Documentation Technique du Moteur ML

Ce document explique l'architecture, les choix techniques et la valeur métier de la partie Intelligence Artificielle de **FleetOpti AI**.

---

## 1. Vision Globale : Le "Pourquoi" ?

L'objectif de FleetOpti AI n'est pas seulement de lister des véhicules, mais d'anticiper les risques opérationnels. Le moteur ML est conçu comme un **organe de décision prédictif** qui transforme les données brutes en informations actionnables :
- **Réduction des coûts** : Passer d'une maintenance réactive (panne) à une maintenance proactive.
- **Conformité & Durabilité** : Estimer précisément l'empreinte carbone pour anticiper les taxes environnementales.
- **Fiabilité Logistique** : Garantir que chaque livraison est sécurisée par l'état de santé du véhicule porteur.

---

## 2. Les Trois Piliers d'Analyse

### A. Maintenance Prédictive (`Maintenance_Required`)
*   **Objectif** : Identifier les véhicules à haut risque de panne avant qu'ils ne quittent l'entrepôt.
*   **Modèle** : `RandomForestClassifier`.
*   **Justification Technique** : Nous utilisons un Random Forest pour sa capacité à gérer des variables mixtes (numériques comme le kilométrage et catégorielles comme le type de batterie) sans nécessiter de normalisation complexe. 
*   **Feature Engineering Clé** : Conversion de `Last_Service_Date` en `Days_Since_Service` pour donner au modèle une notion d'usure temporelle, pas seulement kilométrique.

### B. Estimation de l'Empreinte Carbone
*   **Objectif** : Prédire les émissions réelles de CO2 basées sur les spécifications techniques.
*   **Modèle** : `RandomForestRegressor`.
*   **Valeur Métier** : Permet aux gestionnaires de flotte d'optimiser le renouvellement des véhicules en comparant les émissions théoriques vs prédites en conditions réelles.

### C. Intégrité Logistique
*   **Objectif** : Score de confiance sur une mission de transport.
*   **Méthode** : Corrélation entre la charge du camion (`Actual_Load` / `Load_Capacity`) et les besoins de maintenance.
*   **Action** : Si le modèle détecte une anomalie, la mission peut être réassignée à un autre véhicule plus robuste.

---

## 3. Choix de l'Architecture Technique ("Le Comment")

### Interopérabilité via ONNX
L'application utilise un backend **Java (Spring Boot)**. Pour intégrer des modèles entraînés en **Python**, nous avons choisi le format **ONNX (Open Neural Network Exchange)**.
- **Justification** : Contrairement au format `Pickle`, ONNX est un standard industriel hautement performant qui permet d'exécuter l'inférence en Java sans avoir besoin de Python installé sur le serveur de production.

### Le Système de Métadonnées (`metadata.json`)
Pour que le backend Java puisse utiliser le modèle, il doit savoir exactement comment transformer les données utilisateur :
- **Label Mappings** : Les chaînes de caractères (ex: "Electric", "Broken") sont converties en chiffres via des dictionnaires indexés exportés pendant l'entraînement.
- **Feature Order** : L'ordre des colonnes est figé dans le JSON pour éviter tout décalage d'index lors de l'envoi des données au moteur d'inférence.

---

## 4. Pipeline de Prétraitement (Automation)

Nous avons implémenté une couche `preprocessing.py` centralisée qui :
1.  **Gère les types de dates** : Calcul automatique des deltas temporels (jours restants avant expiration de garantie, jours depuis maintenance).
2.  **Imputation Intelligente** : Remplacement des valeurs manquantes par la médiane numérique, évitant ainsi de perdre des lignes de données précieuses.
3.  **Encodage Robuste** : Utilisation de `LabelEncoder` pour chaque champ textuel, garantissant une traduction parfaite entre le métier et les mathématiques.

---

## 5. Flux d'Intégration (Workflow)

```mermaid
graph LR
    A[Notebook EDA] --> B[Script d'Entraînement Python]
    B --> C[Export ONNX + JSON]
    C --> D[Backend Java / Deep Java Library]
    D --> E[API REST - Dashboard Utilisateur]
```

1.  **Data Scientists** expérimentent dans les `notebooks`.
2.  **Scripts CI/CD** (`train_*.py`) industrialisent l'entraînement.
3.  **Les Artéfacts** (modèles) sont poussés vers le backend.
4.  **L'Utilisateur** reçoit des alertes prédictives en temps réel.

---
*Ce document sert de référence pour comprendre la logique métier et technique derrière l'intelligence de FleetOpti AI.*
