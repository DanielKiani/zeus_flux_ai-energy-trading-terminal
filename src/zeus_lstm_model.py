import torch
import torch.nn as nn
import pytorch_lightning as pl
import logging
import joblib
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ZeusLSTM(nn.Module):
    """
    Production Hybrid LSTM for short-term renewable energy forecasting.
    Uses a Multi-Head architecture with ReLU bounds to predict 5 distinct sectors simultaneously without violating physical limits.
    """
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, output_size: int = 6, num_targets: int = 5, dropout: float = 0.2):
        super(ZeusLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_targets = num_targets 
        
        # Core LSTM Engine
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Stabilization: Layer Normalization prevents internal covariate shift
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Advanced Multi-Head Readout: Deep MLPs per sector instead of shallow linear layers
        self.sector_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.SiLU(),          # Swish activation
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_size),
                nn.ReLU()           # Strict Physics Constraint
            ) for _ in range(num_targets)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        final_time_step = out[:, -1, :]
        
        normed_state = self.layer_norm(final_time_step)
        
        # Pass through the 5 independent sector heads
        sector_predictions = [head(normed_state) for head in self.sector_heads]
        
        # Stack into [batch_size, forecast_horizon, num_targets]
        return torch.stack(sector_predictions, dim=-1)


class ZeusLSTM_model(pl.LightningModule):
    """
    A PyTorch Lightning module integrating the proven Baseline training regime
    (CosineAnnealing, weight_decay) with the Hybrid multi-head architecture.
    """
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 3, output_size: int = 6, num_targets: int = 5, learning_rate: float = 1e-4, weight_decay: float = 1e-3, dropout: float = 0.2, scaler_path: str = "data/processed/target_scaler.save"):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = ZeusLSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            output_size=output_size,
            num_targets=num_targets,
            dropout=dropout
        )
        
        self.loss_fn = nn.L1Loss()
        
        try:
            self.target_scaler = joblib.load(scaler_path)
        except Exception as e:
            logging.warning(f"Could not load multi-dim scaler from {scaler_path}: {e}")
            self.target_scaler = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # --- TRUE REAL-WORLD ACCURACY CALCULATION ---
        if self.target_scaler is not None:
            num_t = self.hparams.num_targets
            # 1. Un-scale back to real Megawatts using CPU numpy
            y_np = y.detach().cpu().numpy().reshape(-1, num_t)
            y_hat_np = y_hat.detach().cpu().numpy().reshape(-1, num_t)
            
            y_mw = self.target_scaler.inverse_transform(y_np).reshape(y.shape)
            y_hat_mw = self.target_scaler.inverse_transform(y_hat_np).reshape(y_hat.shape)
            
            # 2. Sum the 5 sectors into Total Grid Generation
            y_total_mw = np.sum(y_mw, axis=-1)
            y_hat_total_mw = np.sum(y_hat_mw, axis=-1)
            
            # 3. Calculate Apples-to-Apples MAPE
            epsilon = 1.0
            abs_pct_error = np.abs(y_total_mw - y_hat_total_mw) / (np.abs(y_total_mw) + epsilon)
            mape = np.mean(abs_pct_error) * 100
            
            # True Grid Accuracy (e.g., 92.5%)
            acc = max(0.0, 100.0 - mape)
            self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True)
            
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        
        max_epochs = self.trainer.max_epochs if self.trainer.max_epochs else 150
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs, 
            eta_min=1e-6      
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

if __name__ == "__main__":
    try:
        with open("data/processed/data_config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Run prepare_data.py first to generate data_config.json")
        exit(1)

    TOTAL_FEATURES = config["input_size"]
    LOOKBACK_HOURS = config["lookback_hours"]
    FORECAST_HORIZON = config["horizon_hours"]
    NUM_TARGETS = config["num_targets"]
    BATCH_SIZE = 32 
    
    model = ZeusLSTM_model(
        input_size=TOTAL_FEATURES, 
        hidden_size=256, 
        num_layers=3, 
        output_size=FORECAST_HORIZON,
        num_targets=NUM_TARGETS,
        learning_rate=1e-3,     
        weight_decay=1e-3,      
        dropout=0.2             
    )
    
    logging.info(f"Model Initialized: \n{model}")
    
    dummy_input = torch.randn(BATCH_SIZE, LOOKBACK_HOURS, TOTAL_FEATURES)
    output = model(dummy_input)
    
    logging.info(f"Input Tensor Shape: {dummy_input.shape}")
    logging.info(f"Output Tensor Shape: {output.shape}")