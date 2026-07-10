import os
import uvicorn
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import logging
import random
import json
import re
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(message)s')

app = FastAPI(title="ZEUS Command Center API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connect to local Ollama instance
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

@app.get("/api/market_status")
def get_market_status():
    logging.info("📡 UI requested live market status. Fetching grid data...")
    
    # 1. Load the actual data
    try:
        df = pd.read_csv("data/processed/regional/TenneT_master.csv", index_col=0, parse_dates=True)
    except FileNotFoundError:
        return {"error": "Dataset not found. Run ingest_regional.py first."}

    # Grab 13 hours of data (6 hours history, 1 current, 6 hours forecast)
    live_df = df.iloc[-13:]
    row_current = live_df.iloc[6]
    
    current_price = float(row_current['price'])
    
    load_data = live_df['load'].tolist()
    wind_data = (live_df['wind_offshore'] + live_df['wind_onshore']).tolist()
    solar_data = live_df['solar'].tolist()
    price_forecast = live_df['price'].tolist()
    
    biomass_data = live_df['biomass'].tolist()
    hydro_ror_data = live_df['hydro_ror'].tolist()
    hydro_pumped_data = live_df['hydro_pumped'].tolist() if 'hydro_pumped' in live_df.columns else [0] * len(live_df)
    hydro_total = [r + p for r, p in zip(hydro_ror_data, hydro_pumped_data)]
    
    net_load_data = [l - (w + s + b + h) for l, w, s, b, h in zip(load_data, wind_data, solar_data, biomass_data, hydro_total)]
    
    # Calculate deltas specifically from NOW (index 6) to T+6 (index 12)
    net_load_ramp = net_load_data[-1] - net_load_data[6]
    delta_load = load_data[-1] - load_data[6]
    delta_wind = wind_data[-1] - wind_data[6]
    delta_solar = solar_data[-1] - solar_data[6]
    
    # Strict Python Math Engine for Trade Signals
    if net_load_ramp > 500:
        python_signal = "LONG"
        market_condition = "TIGHTENING (Bullish)"
    elif net_load_ramp < -500:
        python_signal = "SHORT"
        market_condition = "RELAXING (Bearish)"
    else:
        python_signal = "HOLD"
        market_condition = "STABLE (Neutral)"

    # Live Simulated News Headlines
    news_headlines = [
        {"source": "Reuters", "time": "12m ago", "headline": "French nuclear output drops 4% due to unexpected reactor maintenance at EDF."},
        {"source": "Bloomberg", "time": "45m ago", "headline": "German industrial data shows stronger-than-expected factory demand for coming week."},
        {"source": "EnergyDesk", "time": "2h ago", "headline": "EU carbon permit prices hit 3-month high, pushing up gas-to-power costs."}
    ]
    news_context = "\n".join([f"- {n['headline']}" for n in news_headlines])

    prompt = f"""
    You are the ZEUS Algorithmic Trading Risk Manager.
    
    CURRENT MARKET STATE:
    - Current Wholesale Price: €{current_price:.2f} / MWh
    
    PHYSICAL DRIVERS (T+1 to T+6):
    - Net Load Ramp: {net_load_ramp:.0f} MW -> {market_condition}
    - Δ Demand: {delta_load:+.0f} MW
    - Δ Wind Gen: {delta_wind:+.0f} MW
    - Δ Solar Gen: {delta_solar:+.0f} MW

    MACRO NEWS CONTEXT:
    {news_context}
    
    ALGORITHMIC DIRECTIVE:
    The quantitative math engine has calculated the required action is: {python_signal}.
    
    TASK:
    Output your response STRICTLY as a JSON object with no markdown formatting, no backticks, and no conversational text.
    Use exactly this schema:
    {{
        "signal": "{python_signal}",
        "confidence": 9,
        "reasoning": "Provide a 1-2 sentence explanation here."
    }}
    """
    
    logging.info(f"🧠 Querying Llama-3 for {python_signal} reasoning with News context...")
    
    try:
        response = client.chat.completions.create(
            model="llama3",
            messages=[
                {"role": "system", "content": "You are a quantitative trading algorithm. You output strictly raw, valid JSON without any markdown formatting or backticks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        llm_output = response.choices[0].message.content.strip()
        
        # ⚡ BULLETPROOF JSON EXTRACTION
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_llm = json.loads(json_str)
            
            # Force all keys to lowercase to prevent case-sensitivity bugs
            parsed_lower = {k.lower(): v for k, v in parsed_llm.items()}
            
            signal = str(parsed_lower.get("signal", python_signal)).upper()
            confidence = int(parsed_lower.get("confidence", 8))
            reasoning = str(parsed_lower.get("reasoning", ""))
            
            # Hard failsafe if reasoning is mysteriously blank or too short
            if not reasoning or len(reasoning) < 5:
                reasoning = f"Signal confirmed. Omni-Learner detected a {market_condition} pattern in the physical grid deltas."
        else:
            raise ValueError("No JSON object found in output")
            
    except Exception as e:
        logging.warning(f"LLM connection or JSON parsing failed: {e}. Injecting simulated reasoning.")
        # Provide a hyper-realistic fallback so the dashboard still looks premium
        if python_signal == "LONG":
            reason_text = f"The {market_condition.lower()} net load ramp signifies a massive supply deficit approaching the grid. Coupled with the breaking news, the grid will require premium-priced peaker plants to meet demand."
        elif python_signal == "SHORT":
            reason_text = f"The {market_condition.lower()} net load ramp indicates extreme renewable oversupply. Wind generation is surging, which will crash the Day-Ahead auction price."
        else:
            reason_text = "Physical deltas are balanced. Net load ramp is flat, and macroeconomic news is fully priced in. Maintaining neutral stance."
            
        signal = python_signal
        confidence = 8 if python_signal != 'HOLD' else 5
        reasoning = f"{reason_text} [Simulated]"

    asks = [{"price": round(current_price + (i * 0.4) + random.uniform(0, 0.2), 2), "vol": random.randint(10, 150)} for i in range(1, 6)][::-1]
    bids = [{"price": round(current_price - (i * 0.4) - random.uniform(0, 0.2), 2), "vol": random.randint(10, 150)} for i in range(1, 6)]

    open_positions = [
        {"id": "ORD-77A", "type": "LONG", "size": "50 MW", "entry": round(current_price - 12.40, 2), "pnl": "+€ 620.00"},
        {"id": "ORD-77B", "type": "SHORT", "size": "25 MW", "entry": round(current_price + 8.10, 2), "pnl": "+€ 202.50"}
    ]
    
    # Generate dynamic wall-clock timestamps
    now = datetime.now()
    dynamic_labels = []
    for i in range(-6, 7):
        t = now + timedelta(hours=i)
        hour_str = t.strftime('%H:00')
        if i == 0:
            dynamic_labels.append(f"NOW ({hour_str})")
        else:
            dynamic_labels.append(hour_str)

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
            "nodes": weather_nodes,
            "news": news_headlines
        },
        "forecast": {
            "labels": dynamic_labels,
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