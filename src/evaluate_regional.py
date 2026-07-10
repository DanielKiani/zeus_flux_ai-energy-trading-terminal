import os
import glob
import torch
import pandas as pd
import numpy as np
import joblib
import logging
import warnings

from train_regional_stacked import (
    prep_regional_data, RegionalLSTMLightning, extract_hidden_states, 
    REGION, TARGETS, EXPERIMENT_NAME, HORIZON
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evaluation():
    logging.info(f"⚡ INITIATING {EXPERIMENT_NAME.upper()} ENSEMBLE BACKTEST ⚡")
    
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
        if not files:
            raise FileNotFoundError(f"No checkpoint found for LSTM '{suffix}'")
        return max(files, key=os.path.getctime)
        
    model_wind = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("wind"))
    model_solar = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("solar"))
    model_e = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("econ"))
    model_h = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("hydro"))
    model_r = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("regime"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    H_wind_test = extract_hidden_states(model_wind, device, X_test)
    H_solar_test = extract_hidden_states(model_solar, device, X_test)
    H_e_test = extract_hidden_states(model_e, device, X_test)
    H_h_test = extract_hidden_states(model_h, device, X_test)
    H_r_test = extract_hidden_states(model_r, device, X_test)
    
    preds_3d = np.zeros((X_test.shape[0], HORIZON, len(TARGETS)))
    
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
    
    preds_t1 = preds_mw[:, 0, :]
    actuals_t1 = actuals_mw[:, 0, :]
    
    preds_load_t1 = preds_t1[:, 0]
    actuals_load_t1 = actuals_t1[:, 0]
    
    preds_gen_t1 = np.sum(preds_t1[:, 1:], axis=1)
    actuals_gen_t1 = np.sum(actuals_t1[:, 1:], axis=1)
    
    preds_net_t1_naive = preds_gen_t1 - preds_load_t1
    actuals_net_t1 = actuals_gen_t1 - actuals_load_t1
    
    try:
        meta_model = joblib.load(f"checkpoints/regional/{REGION}_meta_learner_{EXPERIMENT_NAME}.save")
        
        # ⚡ NEW: Generate the exact same Curtailment & Weather features for Inference
        preds_load = preds_t1[:, 0]
        preds_wind = preds_t1[:, 1] + preds_t1[:, 2] 
        wind_load_ratio = (preds_wind / (preds_load + 1)).reshape(-1, 1)
        
        num_wind_features = Tab_wind_test.shape[1] // HORIZON
        wind_features_t1 = Tab_wind_test[:, :num_wind_features]
        
        X_meta_t1 = np.hstack([preds_t1, wind_features_t1, wind_load_ratio])
        
        # Output all 7 corrected predictions
        preds_t1_corrected = meta_model.predict(X_meta_t1)
        
        preds_load_t1_corrected = preds_t1_corrected[:, 0]
        preds_gen_t1_corrected = np.sum(preds_t1_corrected[:, 1:], axis=1)
        preds_net_t1_corrected = preds_gen_t1_corrected - preds_load_t1_corrected
        
        using_meta_learner = True
    except FileNotFoundError:
        preds_t1_corrected = preds_t1
        preds_net_t1_corrected = preds_net_t1_naive
        using_meta_learner = False
    
    def get_wmape_metrics(pred, actual):
        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual)**2))
        wmape = np.sum(np.abs(pred - actual)) / (np.sum(np.abs(actual)) + 1e-8)
        acc = max(0, 100 - (wmape * 100))
        return mae, rmse, acc

    final_preds_load = preds_load_t1_corrected if using_meta_learner else preds_load_t1
    final_preds_gen = preds_gen_t1_corrected if using_meta_learner else preds_gen_t1
    
    load_mae, load_rmse, load_acc = get_wmape_metrics(final_preds_load, actuals_load_t1)
    gen_mae, gen_rmse, gen_acc = get_wmape_metrics(final_preds_gen, actuals_gen_t1)
    
    net_mae_naive = np.mean(np.abs(preds_net_t1_naive - actuals_net_t1))
    net_mae_corrected = np.mean(np.abs(preds_net_t1_corrected - actuals_net_t1))

    print("\n" + "="*60)
    print(f"⚡ ZEUS {EXPERIMENT_NAME.upper()} ENSEMBLE BACKTEST: {REGION} ⚡")
    print("   [Metrics calculated purely on strict t+1 horizon]")
    print("="*60)
    print(f"🏭 TOTAL RENEWABLE GENERATION:")
    print(f"   Accuracy (wMAPE): {gen_acc:.2f}%  |  MAE: {gen_mae:.0f} MW  |  RMSE: {gen_rmse:.0f} MW")
    print("-" * 60)
    print(f"🏙️ TSO ELECTRICAL LOAD (DEMAND):")
    print(f"   Accuracy (wMAPE): {load_acc:.2f}%  |  MAE: {load_mae:.0f} MW  |  RMSE: {load_rmse:.0f} MW")
    print("-" * 60)
    print(f"⚖️ NET LOAD (SURPLUS / DEFICIT ERROR):")
    if using_meta_learner:
        print(f"   Naive Error Margin    : {net_mae_naive:.0f} MW")
        print(f"   Corrected Error Margin: {net_mae_corrected:.0f} MW 🚀")
    else:
        print(f"   Average Error Margin  : {net_mae_naive:.0f} MW")
    print("=" * 60)
    
    print("Sector-Specific Errors (MAE at t+1):")
    if using_meta_learner:
        print(f"   {'Sector':<15} | {'Base Expert':<13} | {'Omni-Learner'}")
        print("-" * 60)
    else:
        print(f"   {'Sector':<15} | {'Base Expert':<13}")
        print("-" * 60)

    for i in range(len(TARGETS)):
        sector_name = TARGETS[i].replace('_', ' ').title()
        naive_mae = np.mean(np.abs(preds_t1[:, i] - actuals_t1[:, i]))
        if using_meta_learner:
            corrected_mae = np.mean(np.abs(preds_t1_corrected[:, i] - actuals_t1[:, i]))
            diff = naive_mae - corrected_mae
            indicator = "🚀" if diff > 1 else ("⚠️" if diff < -1 else "➖")
            print(f" - {sector_name:<15} | {naive_mae:4.0f} MW        | {corrected_mae:4.0f} MW  {indicator}")
        else:
            print(f" - {sector_name:<15} | {naive_mae:4.0f} MW")
    print("=" * 60)

    print("\n📉 FORECAST DEGRADATION OVER HORIZON (Generation MAE)")
    print("-" * 60)
    for h in range(HORIZON):
        h_preds = np.sum(preds_mw[:, h, 1:], axis=1)
        h_acts = np.sum(actuals_mw[:, h, 1:], axis=1)
        h_mae = np.mean(np.abs(h_preds - h_acts))
        print(f"  Forecast t+{h+1} hr:  {h_mae:.0f} MW MAE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_evaluation()