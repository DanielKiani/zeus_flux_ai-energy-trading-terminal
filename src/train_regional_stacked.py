import os
import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import logging
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

REGION = "TenneT"
LOOKBACK = 72       
HORIZON = 6         
HIDDEN_SIZE = 128   
EPOCHS = 30         
BATCH_SIZE = 64

TARGETS = ['load', 'wind_offshore', 'wind_onshore', 'solar', 'biomass', 'hydro_ror', 'hydro_pumped']
EXPERIMENT_NAME = "v14_spatial"

class RegionalLSTMLightning(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_targets, horizon, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        self.regressor = nn.Linear(hidden_size, num_targets * horizon)
        self.criterion = nn.MSELoss()
        
    def forward(self, x, extract_features=False):
        out, _ = self.lstm(x)
        h_t = out[:, -1, :] 
        if extract_features: return h_t
        return self.regressor(h_t)

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x, extract_features=False)
        loss = self.criterion(outputs, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x, extract_features=False)
        loss = self.criterion(outputs, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

def prep_regional_data():
    file_path = f"data/processed/regional/{REGION}_master.csv"
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    df = df[~df.index.duplicated(keep='first')]
    df = df.resample('1h').interpolate(method='linear').bfill().ffill().fillna(0)
    
    # 1. Cyclical Time
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    # 2. Physics & River Lags
    precip_cols = [c for c in df.columns if 'precipitation' in c]
    if precip_cols:
        df['precip_24h_sum'] = df[precip_cols].sum(axis=1).rolling(24, min_periods=1).sum().fillna(0)
        df['precip_72h_sum'] = df[precip_cols].sum(axis=1).rolling(72, min_periods=1).sum().fillna(0)
    else:
        df['precip_24h_sum'], df['precip_72h_sum'] = 0.0, 0.0
        
    # 3. Economic Pricing
    if 'price' in df.columns:
        shifted_price = df['price'].shift(1).bfill()
        df['price_spread_min_24h'] = shifted_price - shifted_price.rolling(24, min_periods=1).min().fillna(0)
        df['price_spread_max_24h'] = df['price'].rolling(24, min_periods=1).max() - df['price']
        df['price_delta_1h'] = df['price'].diff().fillna(0)
    else:
        df['price_spread_min_24h'], df['price_spread_max_24h'], df['price_delta_1h'] = 0.0, 0.0, 0.0

    # 4. Synthetic Hydro Splitting
    if 'hydro_aggregate' in df.columns and 'hydro_pumped' not in df.columns:
        df['hydro_ror'] = df['hydro_aggregate'].rolling(window=168, min_periods=24).min()
        df['hydro_ror'] = df['hydro_ror'].rolling(window=24).mean().bfill().ffill()
        df['hydro_pumped'] = (df['hydro_aggregate'] - df['hydro_ror']).clip(lower=0)
        
    # ⚡ OPTIMIZATION 3: Clear Sky Index Calculation
    shortwave_cols = [c for c in df.columns if 'shortwave_radiation' in c]
    clearsky_cols = [c for c in df.columns if 'clear_sky_radiation' in c]
    if shortwave_cols and clearsky_cols:
        df['clear_sky_index'] = df[shortwave_cols].mean(axis=1) / (df[clearsky_cols].mean(axis=1) + 1.0)
    else:
        df['clear_sky_index'] = 0.0
        
    lstm_features = list(df.columns)
    
    split_idx_time = int((len(df) - LOOKBACK - HORIZON + 1) * 0.8) + LOOKBACK
    scaler_X, scaler_y = StandardScaler(), StandardScaler()
    
    scaler_X.fit(df[lstm_features].iloc[:split_idx_time])
    scaler_y.fit(df[TARGETS].iloc[:split_idx_time])
    
    X_scaled = scaler_X.transform(df[lstm_features])
    y_scaled = scaler_y.transform(df[TARGETS])
    
    os.makedirs("checkpoints/regional", exist_ok=True)
    joblib.dump(scaler_X, f"checkpoints/regional/{REGION}_scaler_X_{EXPERIMENT_NAME}.save")
    joblib.dump(scaler_y, f"checkpoints/regional/{REGION}_scaler_y_{EXPERIMENT_NAME}.save")
    
    # Feature Routing
    wind_cols = [c for c in df.columns if 'wind_speed' in c]
    solar_cols = [c for c in df.columns if 'radiation' in c] + ['clear_sky_index']
    temp_cols = [c for c in df.columns if 'temperature' in c]
    
    tab_features_wind = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'] + wind_cols
    tab_features_solar = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'] + solar_cols
    tab_features_e = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos'] + temp_cols
    tab_features_h = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'precip_24h_sum', 'precip_72h_sum']
    tab_features_r = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'price', 'price_spread_min_24h', 'price_spread_max_24h', 'price_delta_1h']

    tab_wind_data = df[tab_features_wind].values
    tab_solar_data = df[tab_features_solar].values
    tab_e_data = df[tab_features_e].values
    tab_h_data = df[tab_features_h].values
    tab_r_data = df[tab_features_r].values

    X, Y = [], []
    tab_wind, tab_solar, tab_e, tab_h, tab_r = [], [], [], [], []
    
    for i in range(len(df) - LOOKBACK - HORIZON + 1):
        X.append(X_scaled[i : i + LOOKBACK])
        Y.append(y_scaled[i + LOOKBACK : i + LOOKBACK + HORIZON].flatten())
        
        # ⚡ OPTIMIZATION 1: Maintained exact flattened unrolling across the Horizon
        tab_wind.append(tab_wind_data[i + LOOKBACK : i + LOOKBACK + HORIZON].flatten())
        tab_solar.append(tab_solar_data[i + LOOKBACK : i + LOOKBACK + HORIZON].flatten())
        tab_e.append(tab_e_data[i + LOOKBACK : i + LOOKBACK + HORIZON].flatten())
        tab_h.append(tab_h_data[i + LOOKBACK : i + LOOKBACK + HORIZON].flatten())
        tab_r.append(tab_r_data[i + LOOKBACK : i + LOOKBACK + HORIZON].flatten())
        
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    tab_wind = np.array(tab_wind, dtype=np.float32)
    tab_solar = np.array(tab_solar, dtype=np.float32)
    tab_e = np.array(tab_e, dtype=np.float32)
    tab_h = np.array(tab_h, dtype=np.float32)
    tab_r = np.array(tab_r, dtype=np.float32)
    
    split_idx = int(len(X) * 0.8)
    train_data = (X[:split_idx], Y[:split_idx], tab_wind[:split_idx], tab_solar[:split_idx], tab_e[:split_idx], tab_h[:split_idx], tab_r[:split_idx])
    test_data = (X[split_idx:], Y[split_idx:], tab_wind[split_idx:], tab_solar[split_idx:], tab_e[split_idx:], tab_h[split_idx:], tab_r[split_idx:])
    
    return train_data, test_data, X.shape[2]

def pretrain_lstm_lightning(X_train, Y_train, X_val, Y_val, input_size, target_indices, suffix):
    logging.info(f"⚡ PHASE 2A: Pre-Training {suffix.upper()} LSTM...")
    N_train, N_val = Y_train.shape[0], X_val.shape[0]
    Y_tr_sub = Y_train.reshape(N_train, HORIZON, len(TARGETS))[:, :, target_indices].reshape(N_train, -1)
    Y_val_sub = Y_val.reshape(N_val, HORIZON, len(TARGETS))[:, :, target_indices].reshape(N_val, -1)
    
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(Y_tr_sub)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(Y_val_sub)), batch_size=BATCH_SIZE, shuffle=False)
    
    model = RegionalLSTMLightning(input_size=input_size, hidden_size=HIDDEN_SIZE, num_targets=len(target_indices), horizon=HORIZON)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=True, mode="min")
    
    trainer = pl.Trainer(max_epochs=EPOCHS, accelerator="auto", devices=1, callbacks=[early_stop_callback], enable_progress_bar=True, logger=False)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    ckpt_path = os.path.abspath(f"checkpoints/regional/{REGION}_lstm_{suffix}_{EXPERIMENT_NAME}.ckpt")
    trainer.save_checkpoint(ckpt_path)
    return model, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_hidden_states(model, device, X_data):
    model.to(device)
    model.eval()
    h_t_vectors = []
    with torch.no_grad():
        for i in range(0, len(X_data), 256):
            h_t = model(torch.tensor(X_data[i : i + 256]).to(device), extract_features=True).cpu().numpy()
            h_t_vectors.append(h_t)
    return np.vstack(h_t_vectors)

def train_xgboost_ensemble(H_wind_tr, H_solar_tr, H_e_tr, H_h_tr, H_r_tr, 
                           T_wind_tr, T_solar_tr, T_e_tr, T_h_tr, T_r_tr, Y_train, 
                           H_wind_te, H_solar_te, H_e_te, H_h_te, H_r_te, 
                           T_wind_te, T_solar_te, T_e_te, T_h_te, T_r_te, Y_test):
    logging.info("⚡ PHASE 2B: Training 7 ISOLATED EXPERT MODELS WITH OPTIMIZED HYPERPARAMETERS...")
    
    Y_train_3d = Y_train.reshape(-1, HORIZON, len(TARGETS))
    Y_test_3d = Y_test.reshape(-1, HORIZON, len(TARGETS))
    
    # ⚡ OPTIMIZATION: Inject Optuna's "God-Tier" parameters
    optuna_params = {
        'load': {'n_estimators': 107, 'learning_rate': 0.017459878570673094, 'max_depth': 8, 'subsample': 0.8438451268955002, 'colsample_bytree': 0.9891932796414449, 'min_child_weight': 3, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42},
        'wind_offshore': {'n_estimators': 591, 'learning_rate': 0.02717045591458566, 'max_depth': 4, 'subsample': 0.743089757628234, 'colsample_bytree': 0.6661746093188365, 'min_child_weight': 8, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42},
        'wind_onshore': {'n_estimators': 532, 'learning_rate': 0.11818081334498252, 'max_depth': 3, 'subsample': 0.7260410119934938, 'colsample_bytree': 0.9206768285073078, 'min_child_weight': 3, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42},
        'solar': {'n_estimators': 280, 'learning_rate': 0.08293871958803849, 'max_depth': 5, 'subsample': 0.7356802290866018, 'colsample_bytree': 0.731254305774238, 'min_child_weight': 1, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42},
        'biomass': {'n_estimators': 187, 'learning_rate': 0.013075759105512334, 'max_depth': 4, 'subsample': 0.8790471729848579, 'colsample_bytree': 0.9413976533636776, 'min_child_weight': 5, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42},
        'hydro_ror': {'n_estimators': 534, 'learning_rate': 0.014955127502103555, 'max_depth': 3, 'subsample': 0.7665325905504128, 'colsample_bytree': 0.8603578136212391, 'min_child_weight': 3, 'tree_method': 'hist', 'n_jobs': -1, 'random_state': 42},
        'hydro_pumped': {'n_estimators': 225, 'learning_rate': 0.036976565287131134, 'num_leaves': 86, 'max_depth': 5, 'subsample': 0.9969214290216066, 'colsample_bytree': 0.7155046252142866, 'verbose': -1, 'n_jobs': -1, 'random_state': 42}
    }
    
    preds_stitched = np.zeros((Y_test.shape[0], HORIZON, len(TARGETS)))

    for i, target in enumerate(TARGETS):
        logging.info(f"Training Dedicated Expert for: {target.upper()}")
        
        target_params = optuna_params[target]
        
        # ⚡ OPTIMIZATION 2: Strictly separated arrays for Wind vs. Solar trees
        if target in ['wind_offshore', 'wind_onshore']:
            X_tr, X_te = np.hstack([H_wind_tr, T_wind_tr]), np.hstack([H_wind_te, T_wind_te])
            model = MultiOutputRegressor(XGBRegressor(**target_params))
        elif target == 'solar':
            X_tr, X_te = np.hstack([H_solar_tr, T_solar_tr]), np.hstack([H_solar_te, T_solar_te])
            model = MultiOutputRegressor(XGBRegressor(**target_params))
        elif target in ['load', 'biomass']:
            X_tr, X_te = np.hstack([H_e_tr, T_e_tr]), np.hstack([H_e_te, T_e_te])
            model = MultiOutputRegressor(XGBRegressor(**target_params))
        elif target == 'hydro_ror':
            X_tr, X_te = np.hstack([H_h_tr, T_h_tr]), np.hstack([H_h_te, T_h_te])
            model = MultiOutputRegressor(XGBRegressor(**target_params))
        elif target == 'hydro_pumped':
            X_tr, X_te = np.hstack([H_r_tr, T_r_tr]), np.hstack([H_r_te, T_r_te])
            model = MultiOutputRegressor(lgb.LGBMRegressor(**target_params))
            
        Y_tr = Y_train_3d[:, :, i] 
        model.fit(X_tr, Y_tr)
        
        prefix = 'lgb' if target == 'hydro_pumped' else 'xgb'
        joblib.dump(model, f"checkpoints/regional/{REGION}_{prefix}_{target}_{EXPERIMENT_NAME}.save")
        preds_stitched[:, :, i] = model.predict(X_te)
        
    overall_r2 = r2_score(Y_test, preds_stitched.reshape(-1, HORIZON * len(TARGETS)))
    logging.info(f"✅ God-Tier Pipeline Complete! Meta-Model R^2: {overall_r2:.4f}")

def main():
    train_data, test_data, num_features = prep_regional_data()
    
    # ⚡ OPTIMIZATION 2: Split Weather into Wind and Solar explicit domains
    idx_wind = [1, 2]       # Offshore, Onshore
    idx_solar = [3]         # Solar
    idx_econ = [0, 4]       # Load, Biomass
    idx_hydro = [5]         # Hydro_RoR
    idx_regime = [6]        # Hydro_Pumped
    
    model_wind, dev_wind = pretrain_lstm_lightning(train_data[0], train_data[1], test_data[0], test_data[1], num_features, idx_wind, "wind")
    model_solar, dev_solar = pretrain_lstm_lightning(train_data[0], train_data[1], test_data[0], test_data[1], num_features, idx_solar, "solar")
    model_e, dev_e = pretrain_lstm_lightning(train_data[0], train_data[1], test_data[0], test_data[1], num_features, idx_econ, "econ")
    model_h, dev_h = pretrain_lstm_lightning(train_data[0], train_data[1], test_data[0], test_data[1], num_features, idx_hydro, "hydro")
    model_r, dev_r = pretrain_lstm_lightning(train_data[0], train_data[1], test_data[0], test_data[1], num_features, idx_regime, "regime")
    
    H_wind_tr = extract_hidden_states(model_wind, dev_wind, train_data[0])
    H_solar_tr = extract_hidden_states(model_solar, dev_solar, train_data[0])
    H_e_tr = extract_hidden_states(model_e, dev_e, train_data[0])
    H_h_tr = extract_hidden_states(model_h, dev_h, train_data[0])
    H_r_tr = extract_hidden_states(model_r, dev_r, train_data[0])
    
    H_wind_te = extract_hidden_states(model_wind, dev_wind, test_data[0])
    H_solar_te = extract_hidden_states(model_solar, dev_solar, test_data[0])
    H_e_te = extract_hidden_states(model_e, dev_e, test_data[0])
    H_h_te = extract_hidden_states(model_h, dev_h, test_data[0])
    H_r_te = extract_hidden_states(model_r, dev_r, test_data[0])
    
    train_xgboost_ensemble(
        H_wind_tr, H_solar_tr, H_e_tr, H_h_tr, H_r_tr, 
        train_data[2], train_data[3], train_data[4], train_data[5], train_data[6], train_data[1],
        H_wind_te, H_solar_te, H_e_te, H_h_te, H_r_te, 
        test_data[2], test_data[3], test_data[4], test_data[5], test_data[6], test_data[1]
    )

if __name__ == "__main__":
    main()