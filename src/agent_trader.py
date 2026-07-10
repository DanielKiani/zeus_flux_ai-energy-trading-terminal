import os
import joblib
import numpy as np
import pandas as pd
import logging
from openai import OpenAI  # ⚡ NEW: Import standard client to talk to local Llama3

# We will import the data prep and variables from your existing pipeline
from train_regional_stacked import (
    prep_regional_data, extract_hidden_states, RegionalLSTMLightning,
    REGION, TARGETS, EXPERIMENT_NAME, HORIZON
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🤖 %(message)s')

def generate_market_prompt(t1_preds, t6_preds, current_price):
    """
    Translates raw Megawatt predictions into a financial prompt for the LLM.
    """
    load_t1 = t1_preds[0]
    wind_t1 = t1_preds[1] + t1_preds[2]
    solar_t1 = t1_preds[3]
    net_load_t1 = load_t1 - (wind_t1 + solar_t1 + t1_preds[4] + t1_preds[5] + t1_preds[6])
    
    load_t6 = t6_preds[0]
    wind_t6 = t6_preds[1] + t6_preds[2]
    solar_t6 = t6_preds[3]
    net_load_t6 = load_t6 - (wind_t6 + solar_t6 + t6_preds[4] + t6_preds[5] + t6_preds[6])
    
    # Calculate the ramp and the individual DELTAS
    net_load_ramp = net_load_t6 - net_load_t1
    market_condition = "TIGHTENING (Bullish)" if net_load_ramp > 0 else "RELAXING (Bearish)"
    
    delta_load = load_t6 - load_t1
    delta_wind = wind_t6 - wind_t1
    delta_solar = solar_t6 - solar_t1
    
    prompt = f"""
    You are an elite Quantitative Energy Trader for the European EPEX SPOT market.
    Analyze the following highly accurate ML forecasts for the {REGION} grid and output a trading strategy.
    
    CURRENT MARKET STATE:
    - Current Wholesale Price: €{current_price:.2f} / MWh
    
    T+1 HOUR FORECAST:
    - IMMEDIATE NET LOAD: {net_load_t1:.0f} MW
    
    T+6 HOUR FORECAST:
    - FUTURE NET LOAD: {net_load_t6:.0f} MW
    
    PHYSICAL DRIVERS (T+1 to T+6):
    - Net Load Ramp: {net_load_ramp:.0f} MW -> {market_condition}
    - Δ Demand (Load): {delta_load:+.0f} MW
    - Δ Wind Gen: {delta_wind:+.0f} MW
    - Δ Solar Gen: {delta_solar:+.0f} MW
    
    TASK:
    Based on the Net Load ramp, the physical drivers, and the current price, provide:
    1. SIGNAL: (LONG, SHORT, or HOLD)
    2. CONFIDENCE: (1-10)
    3. REASONING: A 2-sentence professional explanation of the grid physics driving this trade. Do not hallucinate math. Use the physical drivers provided.
    """
    return prompt

def run_trading_desk():
    logging.info("Waking up ZEUS Trading Agent...")
    
    # 1. Load the exact same data as the evaluation script
    _, test_data, _ = prep_regional_data()
    
    try:
        X_test, Y_test_scaled, Tab_w_test, Tab_e_test, Tab_h_test, Tab_r_test = test_data[:6]
    except (KeyError, TypeError):
        vals = list(test_data.values())
        X_test, Y_test_scaled, Tab_w_test, Tab_e_test, Tab_h_test, Tab_r_test = vals[:6]
        
    scaler_y = joblib.load(f"checkpoints/regional/{REGION}_scaler_y_{EXPERIMENT_NAME}.save")
    
    # We will simulate grabbing the "latest" hour of data for a live trade
    latest_X = X_test[-1:]
    latest_Tab_w = Tab_w_test[-1:]
    
    # 2. In a real scenario, we'd pass this through the LSTMs and Omni-Learner
    # For now, we will extract the actual ground truth of the last row to simulate perfect ML output
    latest_actuals_scaled = Y_test_scaled[-1:].reshape(-1, HORIZON, len(TARGETS))
    latest_mw = scaler_y.inverse_transform(latest_actuals_scaled.reshape(-1, len(TARGETS))).reshape(1, HORIZON, len(TARGETS))
    
    t1_preds = latest_mw[0, 0, :]
    t6_preds = latest_mw[0, 5, :]
    
    # 3. Get the latest price (Assuming price is the 5th feature in Tab_r_test)
    # Just a mock current price for the simulation
    current_price = 45.50 

    prompt = generate_market_prompt(t1_preds, t6_preds, current_price)
    
    print("\n" + "="*80)
    print("📡 TRANSMITTING TO LLM TRADING AGENT 📡")
    print("="*80)
    print(prompt)
    print("="*80)
    print("Waiting for LLM integration to parse strategy...\n")

    # ⚡ NEW: Connect to Local Llama 3 via Ollama
    try:
        # Point the standard client to Ollama's default local port
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        
        response = client.chat.completions.create(
            model="llama3", # The exact name of the model in your local Ollama instance
            messages=[
                {"role": "system", "content": "You are a quantitative energy trading algorithm. You do not use pleasantries. Output only the strict requested structured format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Keep it low so the agent is analytical, not creative
            max_tokens=200
        )
        
        print("="*80)
        print("🤖 LLAMA-3 TRADING SIGNAL 🤖")
        print("="*80)
        print(response.choices[0].message.content.strip())
        print("="*80)
        
    except Exception as e:
        logging.error(f"Failed to connect to local Llama 3. Ensure Ollama is running! Error: {e}")

if __name__ == "__main__":
    run_trading_desk()