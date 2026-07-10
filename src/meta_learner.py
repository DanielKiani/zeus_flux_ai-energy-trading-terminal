import os
import torch
import joblib
import logging
import numpy as np
import pandas as pd
import glob
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Import from your main training script
from train_regional_stacked import (
    prep_regional_data, RegionalLSTMLightning, extract_hidden_states,
    REGION, TARGETS, EXPERIMENT_NAME, HORIZON
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("🧠 INITIATING OMNI-LEARNER (CURTAILMENT-AWARE) PIPELINE 🧠")
    
    data_out = prep_regional_data()
    test_data = data_out[1]
    try:
        X_test = test_data[0]
        Y_test_scaled = test_data[1]
        Tab_wind_test = test_data[2]
        Tab_solar_test = test_data[3]
        Tab_e_test = test_data[4]
        Tab_h_test = test_data[5]
        Tab_r_test = test_data[6]
    except (KeyError, TypeError, IndexError):
        vals = list(test_data.values()) if isinstance(test_data, dict) else test_data
        X_test, Y_test_scaled, Tab_wind_test, Tab_solar_test, Tab_e_test, Tab_h_test, Tab_r_test = vals[:7]

    scaler_y = joblib.load(f"checkpoints/regional/{REGION}_scaler_y_{EXPERIMENT_NAME}.save")

    def get_latest_ckpt(suffix):
        pattern = f"checkpoints/regional/{REGION}_lstm_{suffix}_{EXPERIMENT_NAME}*.ckpt"
        files = glob.glob(pattern)
        return max(files, key=os.path.getctime)
    
    logging.info("Loading Frozen Base LSTMs...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_wind = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("wind")).to(device)
    model_solar = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("solar")).to(device)
    model_e = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("econ")).to(device)
    model_h = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("hydro")).to(device)
    model_r = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("regime")).to(device)

    logging.info("Extracting Hidden Contexts...")
    H_wind_test = extract_hidden_states(model_wind, device, X_test)
    H_solar_test = extract_hidden_states(model_solar, device, X_test)
    H_e_test = extract_hidden_states(model_e, device, X_test)
    H_h_test = extract_hidden_states(model_h, device, X_test)
    H_r_test = extract_hidden_states(model_r, device, X_test)

    preds_3d = np.zeros((X_test.shape[0], HORIZON, len(TARGETS)))
    
    logging.info("Generating base predictions from 7 Experts...")
    for i, target in enumerate(TARGETS):
        ckpt_xgb = f"checkpoints/regional/{REGION}_xgb_{target}_{EXPERIMENT_NAME}.save"
        ckpt_lgb = f"checkpoints/regional/{REGION}_lgb_{target}_{EXPERIMENT_NAME}.save"
        model = joblib.load(ckpt_lgb if os.path.exists(ckpt_lgb) else ckpt_xgb)
        
        if target in ['wind_offshore', 'wind_onshore']:
            X_tree = np.hstack([H_wind_test, Tab_wind_test])
        elif target == 'solar':
            X_tree = np.hstack([H_solar_test, Tab_solar_test])
        elif target in ['load', 'biomass']:
            X_tree = np.hstack([H_e_test, Tab_e_test])
        elif target == 'hydro_ror':
            X_tree = np.hstack([H_h_test, Tab_h_test])
        elif target == 'hydro_pumped':
            X_tree = np.hstack([H_r_test, Tab_r_test])
        preds_3d[:, :, i] = model.predict(X_tree)

    preds_scaled = preds_3d.reshape(-1, len(TARGETS))
    actuals_scaled = Y_test_scaled.reshape(-1, len(TARGETS))

    preds_mw = np.maximum(0, scaler_y.inverse_transform(preds_scaled)).reshape(-1, HORIZON, len(TARGETS))
    actuals_mw = scaler_y.inverse_transform(actuals_scaled).reshape(-1, HORIZON, len(TARGETS))

    logging.info("Splitting Test Set into Validation (for Meta-Learner) and Final Test...")
    
    preds_t1 = preds_mw[:, 0, :]
    actuals_t1 = actuals_mw[:, 0, :]
    
    # ⚡ NEW: The Curtailment Proxy
    # We use the Base Experts' own predictions to flag when the grid is overloaded
    preds_load = preds_t1[:, 0]
    preds_wind = preds_t1[:, 1] + preds_t1[:, 2] # Offshore + Onshore
    wind_load_ratio = (preds_wind / (preds_load + 1)).reshape(-1, 1)
    
    # ⚡ NEW: Weather Context
    # Tab_wind_test is flattened over the HORIZON. We slice the t+1 features.
    num_wind_features = Tab_wind_test.shape[1] // HORIZON
    wind_features_t1 = Tab_wind_test[:, :num_wind_features] 
    
    X_meta = np.hstack([preds_t1, wind_features_t1, wind_load_ratio])
    Y_meta = actuals_t1

    split_idx = len(X_meta) // 2
    X_meta_train, X_meta_test = X_meta[:split_idx], X_meta[split_idx:]
    Y_meta_train, Y_meta_test = Y_meta[:split_idx], Y_meta[split_idx:]
    
    # Calculate Naive baseline for comparison
    naive_net_load_test = np.sum(X_meta_test[:, 1:7], axis=1) - X_meta_test[:, 0]
    actual_net_load_test = np.sum(Y_meta_test[:, 1:7], axis=1) - Y_meta_test[:, 0]

    logging.info("Training Multi-Output Omni-Learner with Curtailment Physics...")
    meta_model = MultiOutputRegressor(XGBRegressor(
        n_estimators=100, 
        max_depth=3, # Keep trees shallow to prevent overfitting the biases!
        learning_rate=0.05, 
        random_state=42,
        n_jobs=-1
    ))
    meta_model.fit(X_meta_train, Y_meta_train)

    meta_path = f"checkpoints/regional/{REGION}_meta_learner_{EXPERIMENT_NAME}.save"
    joblib.dump(meta_model, meta_path)
    logging.info(f"💾 Omni-Learner saved successfully to {meta_path}")

    meta_preds_test = meta_model.predict(X_meta_test)
    meta_net_load_test = np.sum(meta_preds_test[:, 1:], axis=1) - meta_preds_test[:, 0]

    naive_mae = np.mean(np.abs(naive_net_load_test - actual_net_load_test))
    meta_mae = np.mean(np.abs(meta_net_load_test - actual_net_load_test))

    print("\n" + "="*60)
    print("🎯 OMNI-LEARNER CURTAILMENT REPORT (t+1 Horizon) 🎯")
    print("="*60)
    print(f"Base Models (Naive Sum) Net Load Error :  {naive_mae:.0f} MW")
    print(f"Omni-Learner Corrected Net Load Error  :  {meta_mae:.0f} MW")
    print("-" * 60)
    if meta_mae < naive_mae:
        print(f"✅ Omni-Learner IMPROVED overall accuracy by {naive_mae - meta_mae:.0f} MW!")
    else:
        print(f"⚠️ Omni-Learner DEGRADED overall accuracy by {meta_mae - naive_mae:.0f} MW.")
    print("=" * 60)

if __name__ == "__main__":
    main()