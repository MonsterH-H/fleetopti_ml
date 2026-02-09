import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import sys
sys.path.append(os.path.abspath('src'))
from preprocessing import load_logistics_data, prepare_splits, export_metadata
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import onnx

def train_logistics_model(csv_path):
    print(f"--- Entrainement Logistique sur {csv_path} ---")
    
    # Prétraitement
    df, encoders = load_logistics_data(csv_path)
    # Cible : Est-ce qu'une maintenance est requise pour assurer la livraison ?
    target = 'Maintenance_Required'
    
    # On retire les colonnes non numériques ou ID avant split
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_splits(df, target_col=target)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Export ONNX
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onx = skl2onnx.convert_sklearn(model, initial_types=initial_type, target_opset=19)
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(os.path.join(model_dir, "logistics_model.onnx"), "wb") as f:
        f.write(onx.SerializeToString())
    
    # Export Metadata
    export_metadata(encoders, feature_names, os.path.join(model_dir, "logistics_metadata.json"))
    
    print("SUCCESS: Modele Logistique exporte : models/logistics_model.onnx")

if __name__ == "__main__":
    DATA_PATH = "data/logistics_dataset_with_maintenance_required.csv"
    if os.path.exists(DATA_PATH):
        train_logistics_model(DATA_PATH)
    else:
        print("ERROR: Fichier " + DATA_PATH + " manquant.")
