import os
import glob
import torch
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
import optuna

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb

# Import your existing pipeline functions
from train_regional_stacked import (
    prep_regional_data, RegionalLSTMLightning, extract_hidden_states, 
    REGION, TARGETS, EXPERIMENT_NAME, HORIZON
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# We only need 20-30 trials per asset to find a massive improvement
N_TRIALS = 30 

def get_latest_ckpt(suffix):
    pattern = f"checkpoints/regional/{REGION}_lstm_{suffix}_{EXPERIMENT_NAME}*.ckpt"
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint found for LSTM '{suffix}' using pattern: {pattern}")
    return max(files, key=os.path.getctime)

def run_optuna_search():
    logging.info("⚡ INITIATING OPTUNA BAYESIAN HYPERPARAMETER SEARCH ⚡")
    
    # 1. Load Data
    logging.info("Loading master datasets...")
    data_out = prep_regional_data()
    train_data = data_out[0]
    test_data = data_out[1]
    
    X_tr, Y_tr_scaled, Tab_wind_tr, Tab_solar_tr, Tab_e_tr, Tab_h_tr, Tab_r_tr = train_data[:7]
    X_te, Y_te_scaled, Tab_wind_te, Tab_solar_te, Tab_e_te, Tab_h_te, Tab_r_te = test_data[:7]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Load Frozen LSTMs & Extract Hidden States
    logging.info("Extracting frozen hidden states from pre-trained LSTMs...")
    model_wind = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("wind")).to(device)
    model_solar = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("solar")).to(device)
    model_e = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("econ")).to(device)
    model_h = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("hydro")).to(device)
    model_r = RegionalLSTMLightning.load_from_checkpoint(get_latest_ckpt("regime")).to(device)
    
    H_wind_tr, H_wind_te = extract_hidden_states(model_wind, device, X_tr), extract_hidden_states(model_wind, device, X_te)
    H_solar_tr, H_solar_te = extract_hidden_states(model_solar, device, X_tr), extract_hidden_states(model_solar, device, X_te)
    H_e_tr, H_e_te = extract_hidden_states(model_e, device, X_tr), extract_hidden_states(model_e, device, X_te)
    H_h_tr, H_h_te = extract_hidden_states(model_h, device, X_tr), extract_hidden_states(model_h, device, X_te)
    H_r_tr, H_r_te = extract_hidden_states(model_r, device, X_tr), extract_hidden_states(model_r, device, X_te)

    # Reshape 3D Targets for fitting
    Y_train_3d = Y_tr_scaled.reshape(-1, HORIZON, len(TARGETS))
    Y_test_3d = Y_te_scaled.reshape(-1, HORIZON, len(TARGETS))

    best_hyperparameters = {}

    def objective(trial, target, X_train_tree, Y_train_tree, X_test_tree, Y_test_tree):
        if target == 'hydro_pumped':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1,
                'n_jobs': -1
            }
            model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
        else:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 600),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tree_method': 'hist',
                'n_jobs': -1,
                'random_state': 42
            }
            model = MultiOutputRegressor(XGBRegressor(**params))
            
        model.fit(X_train_tree, Y_train_tree)
        preds = model.predict(X_test_tree)
        
        # Optimize for Mean Absolute Error (MAE)
        return mean_absolute_error(Y_test_tree, preds)

    # 3. Tune Each Expert
    for i, target in enumerate(TARGETS):
        # ⚡ RESUME FIX: Skip the models that are already tuned!
        if target in ['load', 'wind_offshore', 'wind_onshore', 'solar', 'biomass']:
            logging.info(f"⏭️ Skipping {target.upper()} (Already optimized)")
            continue

        logging.info(f"\n{'='*60}\n🔍 TUNING EXPERT: {target.upper()}\n{'='*60}")
        
        if target in ['wind_offshore', 'wind_onshore']:
            X_tree_tr, X_tree_te = np.hstack([H_wind_tr, Tab_wind_tr]), np.hstack([H_wind_te, Tab_wind_te])
        elif target == 'solar':
            X_tree_tr, X_tree_te = np.hstack([H_solar_tr, Tab_solar_tr]), np.hstack([H_solar_te, Tab_solar_te])
        elif target in ['load', 'biomass']:
            X_tree_tr, X_tree_te = np.hstack([H_e_tr, Tab_e_tr]), np.hstack([H_e_te, Tab_e_te])
        elif target == 'hydro_ror':
            X_tree_tr, X_tree_te = np.hstack([H_h_tr, Tab_h_tr]), np.hstack([H_h_te, Tab_h_te])
        elif target == 'hydro_pumped':
            X_tree_tr, X_tree_te = np.hstack([H_r_tr, Tab_r_tr]), np.hstack([H_r_te, Tab_r_te])
            
        Y_tree_tr = Y_train_3d[:, :, i]
        Y_tree_te = Y_test_3d[:, :, i]

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, target, X_tree_tr, Y_tree_tr, X_tree_te, Y_tree_te), n_trials=N_TRIALS)
        
        logging.info(f"🏆 BEST PARAMS FOR {target.upper()}: {study.best_params}")
        best_hyperparameters[target] = study.best_params
        
    logging.info("\n" + "="*60)
    logging.info("🎯 FINAL OPTIMIZED HYPERPARAMETER DICTIONARY 🎯")
    logging.info("Copy and paste these into your training script!")
    for target, params in best_hyperparameters.items():
        print(f"'{target}': {params},")
    logging.info("="*60)

if __name__ == "__main__":
    run_optuna_search()