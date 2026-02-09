import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocessing import load_maintenance_data, prepare_splits, export_metadata
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import onnx

def train_maintenance_model(csv_path):
    print(f"--- Entrainement Maintenance sur {csv_path} ---")
    
    # Prétraitement
    df, encoders = load_maintenance_data(csv_path)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_splits(df, target_col='Need_Maintenance')
    
    # Modèle Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Export ONNX
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onx = skl2onnx.convert_sklearn(model, initial_types=initial_type)
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(os.path.join(model_dir, "maintenance_model.onnx"), "wb") as f:
        f.write(onx.SerializeToString())
    
    # Export Metadata pour Java
    export_metadata(encoders, feature_names, os.path.join(model_dir, "maintenance_metadata.json"))
    
    print("SUCCESS: Modele Maintenance exporte : models/maintenance_model.onnx")

if __name__ == "__main__":
    DATA_PATH = "data/vehicle_maintenance_data.csv"
    if os.path.exists(DATA_PATH):
        train_maintenance_model(DATA_PATH)
    else:
        print("ERROR: Fichier " + DATA_PATH + " manquant.")
