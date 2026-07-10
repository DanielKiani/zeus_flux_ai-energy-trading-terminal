import os
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import logging
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')

app = FastAPI(title="ZEUS Command Center API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your local HTML file to talk to this server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

@app.get("/api/market_status")
def get_market_status():
    logging.info("📡 UI requested live market status. Fetching grid data...")
    
    # 1. Load the latest "live" data (simulated from our master CSV)
    try:
        df = pd.read_csv("data/processed/regional/TenneT_master.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return {"error": "Dataset not found. Run ingest_regional.py first."}

    # Slice the current hour (T) and the 6-hour forecast (T+1 to T+6)
    live_df = df.iloc[-7:]
    
    row_current = live_df.iloc[0]
    forecast_df = live_df.iloc[1:]
    
    current_price = float(row_current['price'])
    
    # 2. Extract arrays for the UI Chart
    labels = forecast_df.index.strftime('T+%H:%M').tolist()
    load_data = forecast_df['load'].tolist()
    wind_data = (forecast_df['wind_offshore'] + forecast_df['wind_onshore']).tolist()
    solar_data = forecast_df['solar'].tolist()
    price_forecast = forecast_df['price'].tolist()
    
    # --- ADDING GENERATION STACK DATA ---
    biomass_data = forecast_df['biomass'].tolist()
    hydro_ror_data = forecast_df['hydro_ror'].tolist()
    if 'hydro_pumped' in forecast_df.columns:
        hydro_pumped_data = forecast_df['hydro_pumped'].tolist()
    else:
        hydro_pumped_data = [0] * len(forecast_df)
    
    hydro_total = [r + p for r, p in zip(hydro_ror_data, hydro_pumped_data)]
    
    # Calculate Net Load
    net_load_data = [l - (w + s + b + h) for l, w, s, b, h in zip(load_data, wind_data, solar_data, biomass_data, hydro_total)]
    
    # 3. Calculate Deltas and Enforce Python Logic (The Quant Fix)
    net_load_ramp = net_load_data[-1] - net_load_data[0]
    delta_load = load_data[-1] - load_data[0]
    delta_wind = wind_data[-1] - wind_data[0]
    delta_solar = solar_data[-1] - solar_data[0]
    
    if net_load_ramp > 500:
        python_signal = "LONG"
        market_condition = "TIGHTENING (Bullish)"
    elif net_load_ramp < -500:
        python_signal = "SHORT"
        market_condition = "RELAXING (Bearish)"
    else:
        python_signal = "HOLD"
        market_condition = "STABLE (Neutral)"

    # 4. Query Llama-3 for Reasoning
    prompt = f"""
    You are the ZEUS Algorithmic Trading Risk Manager.
    
    CURRENT MARKET STATE:
    - Current Wholesale Price: €{current_price:.2f} / MWh
    
    GRID FORECAST (T+1 to T+6):
    - Net Load Ramp: {net_load_ramp:.0f} MW -> {market_condition}
    - Δ Demand: {delta_load:+.0f} MW
    - Δ Wind Gen: {delta_wind:+.0f} MW
    - Δ Solar Gen: {delta_solar:+.0f} MW
    
    ALGORITHMIC DIRECTIVE:
    Based on strict quantitative thresholds, the math engine has determined the required action is: {python_signal}.
    
    TASK:
    1. Output strictly: SIGNAL: {python_signal}
    2. Output CONFIDENCE: (1-10)
    3. Output REASONING: Explain exactly WHY this {python_signal} makes sense physically based on the grid deltas.
    """
    
    logging.info(f"🧠 Querying Llama-3 for {python_signal} reasoning...")
    
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a quantitative trading algorithm. Output only the requested format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        llm_output = response.choices[0].message.content.strip()
    except Exception as e:
        llm_output = f"SIGNAL: {python_signal}\nCONFIDENCE: 9\nREASONING: LLM Connection Failed. Relying on raw Python math engine."

    signal = python_signal
    confidence = 8
    reasoning = llm_output

    for line in llm_output.split('\n'):
        if "SIGNAL:" in line: signal = line.replace("SIGNAL:", "").strip()
        if "CONFIDENCE:" in line: 
            try: confidence = int(line.replace("CONFIDENCE:", "").strip())
            except ValueError: pass
        if "REASONING:" in line: reasoning = line.replace("REASONING:", "").strip()

    # Generate a live simulated Order Book (EPEX SPOT Depth)
    asks = [{"price": round(current_price + (i * 0.4) + random.uniform(0, 0.2), 2), "vol": random.randint(10, 150)} for i in range(1, 6)][::-1]
    bids = [{"price": round(current_price - (i * 0.4) - random.uniform(0, 0.2), 2), "vol": random.randint(10, 150)} for i in range(1, 6)]

    # --- NEW: Generate Simulated Trade Blotter Data ---
    open_positions = [
        {"id": "ORD-77A", "type": "LONG", "size": "50 MW", "entry": round(current_price - 12.40, 2), "mark": round(current_price, 2), "pnl": "+€ 620.00"},
        {"id": "ORD-77B", "type": "SHORT", "size": "25 MW", "entry": round(current_price + 8.10, 2), "mark": round(current_price, 2), "pnl": "+€ 202.50"}
    ]

    # --- NEW: Generate Spatial Weather Nodes ---
    weather_nodes = [
        {"id": "north_sea", "name": "North Sea Offshore", "top": "15%", "left": "25%", "color": "red", "delta": "-1,850 MW", "pulse": True},
        {"id": "schleswig", "name": "Schleswig Coast", "top": "20%", "left": "50%", "color": "red", "delta": "-1,200 MW", "pulse": True},
        {"id": "lower_saxony_n", "name": "Bremen Corridor", "top": "35%", "left": "35%", "color": "yellow", "delta": "-850 MW", "pulse": False},
        {"id": "lower_saxony_s", "name": "Hannover Corridor", "top": "50%", "left": "45%", "color": "yellow", "delta": "-450 MW", "pulse": False},
        {"id": "hesse", "name": "Hesse Central", "top": "65%", "left": "40%", "color": "green", "delta": "-100 MW", "pulse": False},
        {"id": "bavaria_n", "name": "Northern Bavaria", "top": "75%", "left": "65%", "color": "green", "delta": "+20 MW", "pulse": False},
        {"id": "bavaria_s", "name": "Munich / Alps", "top": "90%", "left": "70%", "color": "green", "delta": "-20 MW", "pulse": False},
    ]

    payload = {
        "status": "success",
        "market": {
            "current_price": current_price,
            "trend": market_condition.split(" ")[0],
            "net_load_ramp": net_load_ramp,
            "delta_load": delta_load,
            "delta_wind": delta_wind,
            "delta_solar": delta_solar,
            "intraday_spread": round(random.uniform(-4.50, 4.50), 2),
            "order_book": {"asks": asks, "bids": bids},
            "pnl": "+€ 2,450.00",
            "positions": open_positions,
            "nodes": weather_nodes
        },
        "forecast": {
            "labels": labels,
            "load": load_data,
            "wind": wind_data,
            "solar": solar_data,
            "biomass": biomass_data,
            "hydro": hydro_total,
            "net_load": net_load_data,
            "price": price_forecast
        },
        "agent": {
            "signal": signal.upper(),
            "confidence": confidence,
            "reasoning": reasoning
        }
    }
    
    return payload

if __name__ == "__main__":
    logging.info("🚀 Starting ZEUS Backend Server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)