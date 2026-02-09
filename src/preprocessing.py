import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import os

def load_maintenance_data(file_path, encode=True):
    """Prépare les données pour la maintenance prédictive."""
    df = pd.read_csv(file_path)
    le_dict = {}

    # Gestion des dates
    if 'Last_Service_Date' in df.columns:
        df['Last_Service_Date'] = pd.to_datetime(df['Last_Service_Date'], errors='coerce')
        df['Days_Since_Service'] = (pd.Timestamp.now() - df['Last_Service_Date']).dt.days
        df = df.drop(columns=['Last_Service_Date'])
    
    if 'Warranty_Expiry_Date' in df.columns:
        df['Warranty_Expiry_Date'] = pd.to_datetime(df['Warranty_Expiry_Date'], errors='coerce')
        df['Days_Until_Expiry'] = (df['Warranty_Expiry_Date'] - pd.Timestamp.now()).dt.days
        df = df.drop(columns=['Warranty_Expiry_Date'])

    if encode:
        cat_cols = ['Vehicle_Model', 'Maintenance_History', 'Fuel_Type', 
                    'Transmission_Type', 'Owner_Type', 'Tire_Condition', 
                    'Brake_Condition', 'Battery_Status', 'Vehicle_Type']
        for col in cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le

    # Remplissage des valeurs manquantes numériques
    df = df.fillna(df.median(numeric_only=True))
    return df, le_dict

def load_co2_data(file_path, encode=True):
    """Prépare les données pour le calcul carbone."""
    df = pd.read_csv(file_path)
    le_dict = {}
    if encode:
        for col in ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
    return df, le_dict

def load_logistics_data(file_path, encode=True):
    """Prépare les données pour l'optimisation logistique."""
    df = pd.read_csv(file_path)
    # Feature engineering : Ratio de charge
    if 'Actual_Load' in df.columns and 'Load_Capacity' in df.columns:
        df['Load_Utilization'] = df['Actual_Load'] / df['Load_Capacity']
    
    # Encodage des conditions
    le_dict = {}
    if encode:
        for col in ['Weather_Conditions', 'Road_Conditions', 'Vehicle_Type', 'Maintenance_History']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                le_dict[col] = le
    return df, le_dict

def load_telematics_data(file_path):
    """Prépare les données de télématique."""
    df = pd.read_csv(file_path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    return df

def export_metadata(encoders, features, output_path="models/metadata.json"):
    """Exporte les mappings des encoders et l'ordre des colonnes pour Java."""
    metadata = {
        "features": features,
        "mappings": {}
    }
    for col, le in encoders.items():
        metadata["mappings"][col] = {str(label): int(i) for i, label in enumerate(le.classes_)}
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"SUCCESS: Metadonnees exportees vers {output_path}")

def prepare_splits(df, target_col):
    """Split générique Train/Test avec retour de scaler et feature_names."""
    y = df[target_col]
    X = df.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number])
    # Supprimer les IDs connus qui ne sont pas des features
    X = X.drop(columns=['Vehicle_ID', 'deviceId', 'timeMili', 'id', 'ID'], errors='ignore')
    
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names
