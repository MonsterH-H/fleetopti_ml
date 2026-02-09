# 🎓 Masterclass : Architecture ML pour l'Intelligence de Flotte (Fleet Intelligence)

Ce cours ultra-détaillé vous guide à travers chaque étape de la construction du moteur d'intelligence de **FleetOpti AI**. Nous allons explorer les mathématiques, le code et les stratégies de déploiement industriel.

---

## 📑 Sommaire Détaillé
1.  **Module 1 : Data Preprocessing & Feature Engineering** (La Magie du Temps)
2.  **Module 2 : EDA & Analyse Statistique** (Détecter les Signaux Faibles)
3.  **Module 3 : Algorithmes & Évaluation** (Pourquoi le Random Forest ?)
4.  **Module 4 : Architecture de Déploiement ONNX** (Le Pont Python-Java)
5.  **Module 5 : Industrialisation & Maintenance** (Automation & Robustesse)

---

## 🛠️ Module 1 : Préparation & Ingénierie des Données

Le succès d'un modèle de maintenance ne dépend pas de l'algorithme, mais de la manière dont vous "nourrissez" la donnée.

### 1.1 La Transformation Temporelle (Time-Delta Engineering)
Les caractéristiques de date (`2023-01-15`) sont inexploitables par un modèle mathématique. Nous devons les convertir en **grandeurs scalaires**.
*   **Logique** : Ce qui importe n'est pas *quand* le véhicule a été révisé, mais *depuis combien de temps* il l'a été.
*   **Implémentation** :
    ```python
    # On calcule la différence entre 'maintenant' et la date de service
    # .dt.days transforme l'objet Timedelta en un entier (int)
    df['Days_Since_Service'] = (pd.Timestamp.now() - pd.to_datetime(df['Last_Service_Date'])).dt.days
    ```
*   **Pourquoi ?** Un entier permet au modèle de créer des règles comme : `if Days_Since_Service > 180 then Risk += 20%`.

### 1.2 Encodage Catégoriel Stratégique
Les modèles Scikit-Learn ne comprennent que les nombres de type `float` ou `int`. 
*   **Label Encoding** : On assigne un chiffre unique à chaque étiquette (ex: `Truck=0`, `Van=1`).
*   **Précaution Critique** : Toujours utiliser `.astype(str)` avant l'encodage pour éviter les erreurs si une colonne contient un mélange de types (ex: `NaN` et `Strings`).

### 1.3 Nettoyage & Imputation
*   **Le problème des NaNs** (Valeurs manquantes) : Une ligne avec un trou peut faire planter l'entraînement.
*   **La solution Médiane** : Contrairement à la moyenne, la médiane n'est pas influencée par les valeurs extrêmes (ex: un camion accidenté avec 1 000 000 km).
    ```python
    df = df.fillna(df.median(numeric_only=True))
    ```

---

## 📊 Module 2 : Analyse Exploratoire (EDA) & Corrélations

L'EDA permet de "voir" la physique du problème avant de lancer les calculs.

### 2.1 Matrice de Corrélation
Nous cherchons à quantifier le lien entre nos variables et la cible (`Need_Maintenance`).
*   **Action** : `df.corr(numeric_only=True)`. 
*   **Interprétation** : 
    *   **+1.0** : Corrélation positive parfaite (si X monte, Y monte).
    *   **-1.0** : Corrélation négative (si l'âge monte, la fiabilité descend).
*   **Visualisation** : Utilisez `sns.heatmap` avec une palette divergente (`RdBu`) pour repérer instantanément les variables critiques (Kilométrage, Âge, Nombre d'incidents signalés).

### 2.2 Analyse de Densité (KDE Plots)
Les graphiques de densité permettent de voir si deux populations (Vehicules OK vs Véhicules en Panne) se séparent bien sur une variable donnée.
*   **Exemple** : Si les pics de densité du kilométrage pour les véhicules "OK" et "Panne" sont trop proches, le kilométrage seul ne suffira pas à prédire la panne. Il faudra combiner avec l'âge.

---

## 🤖 Module 3 : Modélisation Prédictive & Évaluation

### 3.1 Pourquoi le Random Forest (Forêt Aléatoire) ?
Le Random Forest est un algorithme d'**Ensemble Learning** (Bagging). Il crée des centaines d'arbres de décision et vote pour le résultat final.
1.  **Non-linéarité** : Il capte des relations complexes que la régression linéaire ignore.
2.  **Importance des variables** : Après l'entraînement, on peut extraire `feature_importances_` pour dire au client : *"C'est le kilométrage qui pèse 60% dans votre risque de panne"*.

### 3.2 Métriques de Succès
*   **Pour la Classification (Maintenace)** :
    *   **Précision** : "Sur tous mes signalements de panne, combien étaient vrais ?"
    *   **Rappel (Recall)** : "Sur toutes les pannes réelles, combien en ai-je détectées ?" (Crucial en maintenance pour ne rien rater).
*   **Pour la Régression (CO2)** :
    *   **R² (Coefficient de détermination)** : Pourcentage de la variance expliqué par le modèle (Objectif > 0.90).
    *   **MAE** : Erreur moyenne en grammes de CO2 (ex: "Le modèle se trompe en moyenne de 5g").

---

## 🚀 Module 4 : Le Pont Industriel (Python vers Java avec ONNX)

C'est l'étape la plus complexe : intégrer l'IA dans une application de production.

### 4.1 Qu'est-ce que l'ONNX ?
**Open Neural Network Exchange** est un format binaire universel. Il permet de "geler" l'intelligence du modèle Python pour qu'elle soit exécutable par un moteur ultra-rapide en **Java C++ ou C#**.

### 4.2 Le Pipeline d'Export
```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# 1. Définir le format d'entrée (Nombre de colonnes fixes)
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]

# 2. Convertir
onx = convert_sklearn(model, initial_types=initial_type)

# 3. Sauvegarder
with open("maintenance_model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

### 4.3 Le rôle vital des Métadonnées (`metadata.json`)
Le fichier ONNX est une boîte noire mathématique. Il ne sait pas que la colonne 0 est le "Kilométrage". 
*   **Metadata** : Nous exportons un fichier JSON contenant l'ordre des colonnes et les dictionnaires de LabelEncoding. Sans ce fichier, le backend Java ne peut pas envoyer les bonnes données au modèle.

---

## 🧹 Module 5 : Bonnes Pratiques de Production (MLOps)

### 5.1 Environnement de Développement Cycle-Court
L'utilisation de `%load_ext autoreload` est indispensable. Elle permet au Data Scientist de coder dans `preprocessing.py` et de tester immédiatement dans son Notebook sans recharger tout l'environnement de données (gain de productivité massif).

### 5.2 Robustesse des Prédictions (`numeric_only=True`)
Dans les nouvelles versions de Pandas, les calculs statistiques sur des dataframes mixtes (textes + nombres) lèvent des erreurs. Utiliser explicitement `numeric_only=True` garantit que votre pipeline ne cassera pas si une nouvelle colonne textuelle est ajoutée au dataset.

### 5.3 Scaling (Normalisation)
Nous utilisons `StandardScaler` pour que toutes les variables soient sur la même échelle (moyenne=0, écart-type=1). 
*   **Règle d'or** : Le `scaler` doit être entraîné sur le `Train set` et appliqué tel quel sur le `Test set` (et en production) pour éviter toute fuite de données (Data Leakage).

---

## 📝 Conclusion : Les 3 Piliers du Succès
1.  **Features over Algos** : Passer 80% du temps sur le prétraitement (Module 1).
2.  **Traçabilité absolue** : Si vous changez une règle dans Python, elle doit être reflétée dans le JSON pour le backend Java.
3.  **Validation Métier** : Un score de 99% suspect est souvent signe de "Data Leakage" (ex: inclure la cause de la panne dans les données d'entraînement).

---
**Félicitations !** Vous maîtrisez maintenant l'architecture complète du moteur prédictif de **FleetOpti AI**.
