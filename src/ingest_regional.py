import os
import requests
import pandas as pd
import logging
import time
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODULES = {
    "load": 410,
    "wind_offshore": 1225,
    "hydro_ror": 1226,
    "biomass": 4066,
    "wind_onshore": 4067,
    "solar": 4068,
    "hydro_pumped": 4070
}

# ⚡ OPTIMIZATION 4: 7-Node Spatial Grid to track moving weather fronts
WEATHER_CLUSTERS = {
    "TenneT": {
        "north_sea": {"lat": 54.5, "lon": 6.5},         # Offshore
        "schleswig": {"lat": 54.3, "lon": 9.5},         # Northern Coast
        "lower_saxony_n": {"lat": 53.1, "lon": 9.2},    # Bremen Corridor
        "lower_saxony_s": {"lat": 52.0, "lon": 9.9},    # Hannover Corridor
        "hesse": {"lat": 51.0, "lon": 9.5},             # Central
        "bavaria_n": {"lat": 49.8, "lon": 10.9},        # Northern Bavaria
        "bavaria_s": {"lat": 48.1, "lon": 11.5}         # Munich/Alps
    }
}

def fetch_smard_history(module_name: str, module_id: int, region_code: str = "DE", weeks: int = 52) -> pd.DataFrame:
    logging.info(f"Fetching SMARD history for {module_name} (ID: {module_id}) in {region_code}...")
    url = f"https://www.smard.de/app/chart_data/{module_id}/{region_code}/index_hour.json"
    
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        
        timestamps = res.json().get("timestamps", [])
        timestamps = timestamps[-weeks:] 
        
        data = []
        for ts in timestamps:
            data_url = f"https://www.smard.de/app/chart_data/{module_id}/{region_code}/{module_id}_{region_code}_hour_{ts}.json"
            r = requests.get(data_url, timeout=10)
            if r.status_code == 200:
                data.extend(r.json().get("series", []))
            time.sleep(0.1) 
            
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data, columns=['time', module_name])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df = df.resample('1h').mean()
        return df
        
    except Exception as e:
        logging.error(f"Failed to fetch {module_name}: {e}")
        return pd.DataFrame()

def fetch_weather_history(cluster_name: str, coords: dict, days_back: int = 365) -> pd.DataFrame:
    logging.info(f"Fetching Open-Meteo Archive + Live for {cluster_name}...")
    
    hourly_vars = "temperature_2m,wind_speed_10m,wind_speed_100m,direct_radiation,diffuse_radiation,shortwave_radiation,precipitation"
    
    # 1. Fetch deep history from Archive API
    archive_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    end_date = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    
    archive_params = {
        "latitude": coords['lat'],
        "longitude": coords['lon'],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": "UTC"
    }
    
    # 2. Fetch recent history + today from Forecast API (covers the 5-day archive lag)
    forecast_url = "https://api.open-meteo.com/v1/forecast"
    forecast_params = {
        "latitude": coords['lat'],
        "longitude": coords['lon'],
        "past_days": 7,
        "forecast_days": 1,
        "hourly": hourly_vars,
        "timezone": "UTC"
    }

    def fetch_api(url, params):
        for attempt in range(5):
            res = requests.get(url, params=params, timeout=30)
            if res.status_code == 429:
                logging.warning(f"Rate limited. Sleeping {2 ** attempt}s...")
                time.sleep(2 ** attempt)
                continue
            res.raise_for_status()
            df = pd.DataFrame(res.json()['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            return df
        return pd.DataFrame()

    df_archive = fetch_api(archive_url, archive_params)
    df_forecast = fetch_api(forecast_url, forecast_params)

    # Stitch them together and drop overlapping days
    df_combined = pd.concat([df_archive, df_forecast])
    df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
    
    df_combined.columns = [f"{c}_{cluster_name}" for c in df_combined.columns]
    return df_combined

def build_regional_dataset(region: str):
    logging.info(f"=== Assembling V14 Master Dataset for {region} ===")
    
    smard_dfs = []
    for name, mod_id in MODULES.items():
        df = fetch_smard_history(name, mod_id, region_code=region)
        if not df.empty:
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            smard_dfs.append(df)
            
    grid_df = pd.concat(smard_dfs, axis=1).sort_index()
    
    weather_dfs = []
    if region in WEATHER_CLUSTERS:
        for cluster_name, coords in WEATHER_CLUSTERS[region].items():
            w_df = fetch_weather_history(cluster_name, coords)
            if not w_df.empty:
                if w_df.index.tz is None:
                    w_df.index = w_df.index.tz_localize('UTC')
                weather_dfs.append(w_df)
    
    weather_df = pd.concat(weather_dfs, axis=1)
    
    price_df = fetch_smard_history("price", 4169, region_code="DE-LU")
    if not price_df.empty and price_df.index.tz is None:
        price_df.index = price_df.index.tz_localize('UTC')
        
    master_df = pd.merge(grid_df, weather_df, left_index=True, right_index=True, how='inner')
    master_df = pd.merge(master_df, price_df, left_index=True, right_index=True, how='inner')
    
    master_df = master_df[~master_df.index.duplicated(keep='first')]
    master_df = master_df.resample('1h').interpolate(method='linear').bfill().ffill().fillna(0)
    
    # Smarter Night Mask based on physical radiation, not just hours
    if 'solar' in master_df.columns:
        shortwave_cols = [c for c in master_df.columns if 'shortwave_radiation' in c]
        if shortwave_cols:
            night_mask = master_df[shortwave_cols].sum(axis=1) <= 5.0
            master_df.loc[night_mask, 'solar'] = 0.0
        
    os.makedirs("data/processed/regional", exist_ok=True)
    out_path = f"data/processed/regional/{region}_master.csv"
    master_df.to_csv(out_path)
    logging.info(f"✅ V14 Master Dataset saved to {out_path}")

if __name__ == "__main__":
    build_regional_dataset("TenneT")