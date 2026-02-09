import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from preprocessing import load_co2_data, prepare_splits, export_metadata
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
import onnx

def train_co2_model(csv_path):
    print(f"--- Entrainement Empreinte Carbone sur {csv_path} ---")
    
    # Prétraitement
    df, encoders = load_co2_data(csv_path)
    # On cible 'CO2 Emissions(g/km)'
    target = 'CO2 Emissions(g/km)'
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_splits(df, target_col=target)
    
    # Modèle de régression pour prédire une valeur continue
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f} g/km")
    
    # Export ONNX
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onx = skl2onnx.convert_sklearn(model, initial_types=initial_type, target_opset=19)
    
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    with open(os.path.join(model_dir, "co2_model.onnx"), "wb") as f:
        f.write(onx.SerializeToString())
        
    # Export Metadata
    export_metadata(encoders, feature_names, os.path.join(model_dir, "co2_metadata.json"))
    
    print("SUCCESS: Modele CO2 exporte : models/co2_model.onnx")

if __name__ == "__main__":
    DATA_PATH = "data/CO2 Emissions_Canada.csv"
    if os.path.exists(DATA_PATH):
        train_co2_model(DATA_PATH)
    else:
        print("ERROR: Fichier " + DATA_PATH + " manquant.")
