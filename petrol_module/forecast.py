import pandas as pd
import numpy as np
from datetime import timedelta

def generate_7day_forecast(
    best_model,
    latest_df,
    feature_cols,
    base_price=None,
    base_date=None,
    market_scenario="⚖️ Dynamic Market Volatility (Fluctuating Up & Down)",
    crude_shock_pct=0.0
):
    """Generates an iterative 7-day future forecast starting from the latest available date or live price.
    
    Supports realistic bi-directional price movements (both downward drops and upward hikes) via:
    1. Continuous macroeconomic transmission (Brent Crude trajectory and USD/INR dynamics).
    2. Market regime simulation:
       - '📉 Bearish Crude / Price Reduction Cycle': crude softening yields retail price reductions (▼).
       - '⚖️ Dynamic Market Volatility': realistic day-to-day fluctuations (both ▼ and ▲).
       - '🎯 OMC Rate Pause (Regulated Stability)': price freeze within ±0.02.
       - '📈 Bullish Crude Squeeze': crude rally driving upward adjustments (▲).
    3. User-defined crude oil shock slider (-10% to +10%).
    """
    history_df = latest_df.copy().sort_values('Date').reset_index(drop=True)
    latest_date = pd.to_datetime(base_date) if base_date else history_df['Date'].max()
    latest_actual_price = float(base_price) if base_price is not None else float(history_df['Petrol_Price'].iloc[-1])
    
    # Smoothly align recent history around base_price so artificial multi-year gaps don't distort lag differences
    if base_price is not None:
        price_offset = latest_actual_price - float(history_df['Petrol_Price'].iloc[-1])
        history_df['Petrol_Price'] = history_df['Petrol_Price'] + price_offset

    forecast_records = []
    current_history = history_df[['Date', 'Petrol_Price', 'Crude_Oil_USD', 'USD_INR']].copy()

    # Base international benchmarks
    last_crude = float(current_history['Crude_Oil_USD'].iloc[-1])
    last_usd = float(current_history['USD_INR'].iloc[-1])

    # Preset macro shock profiles for the 7-day roll forward
    if "Bearish" in market_scenario:
        # Crude oil softens: daily negative changes with OMC price cuts
        daily_crude_deltas = [-0.55, -0.65, -0.40, -0.60, -0.45, -0.35, -0.50]
        drift_bias = -0.025
    elif "Volatility" in market_scenario:
        # Bidirectional fluctuations (real-world daily fuel movements)
        daily_crude_deltas = [-0.50, -0.35, +0.40, -0.40, +0.30, -0.35, +0.15]
        drift_bias = 0.00
    elif "Pause" in market_scenario:
        # Rate freeze / regulated pause
        daily_crude_deltas = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        drift_bias = 0.00
    else:  # Bullish
        # Crude rally / upward cost pressures
        daily_crude_deltas = [+0.45, +0.55, +0.35, +0.50, +0.40, +0.45, +0.35]
        drift_bias = +0.020

    # Apply additional user-defined crude shock (scaled across 7 days)
    user_daily_crude_shift = (last_crude * (crude_shock_pct / 100.0)) / 7.0

    current_crude = last_crude
    current_usd = last_usd

    for day in range(1, 8):
        next_date = latest_date + timedelta(days=day)
        
        # Evolve international crude benchmark
        crude_step = daily_crude_deltas[day - 1] + user_daily_crude_shift
        current_crude = round(max(30.0, current_crude + crude_step), 2)
        
        petrol_prices = current_history['Petrol_Price'].values
        crude_prices = current_history['Crude_Oil_USD'].values
        usd_rates = current_history['USD_INR'].values
        
        p_lag_1 = petrol_prices[-1]
        p_lag_2 = petrol_prices[-2] if len(petrol_prices) >= 2 else p_lag_1
        p_lag_7 = petrol_prices[-7] if len(petrol_prices) >= 7 else petrol_prices[0]
        
        c_lag_1 = crude_prices[-1]
        c_lag_2 = crude_prices[-2] if len(crude_prices) >= 2 else c_lag_1
        c_lag_7 = crude_prices[-7] if len(crude_prices) >= 7 else crude_prices[0]
        
        u_lag_1 = usd_rates[-1]
        u_lag_2 = usd_rates[-2] if len(usd_rates) >= 2 else u_lag_1
        
        roll_std_7 = float(np.std(petrol_prices[-7:]))
        
        row_dict = {
            'DayOfWeek': next_date.dayofweek,
            'Month': next_date.month,
            'Quarter': (next_date.month - 1) // 3 + 1,
            'Petrol_Diff_1': float(p_lag_1 - p_lag_2),
            'Petrol_Diff_7': float(p_lag_1 - p_lag_7),
            'Crude_Lag_1': float(c_lag_1),
            'Crude_Diff_1': float(c_lag_1 - c_lag_2),
            'Crude_Diff_7': float(c_lag_1 - c_lag_7),
            'USD_Lag_1': float(u_lag_1),
            'USD_Diff_1': float(u_lag_1 - u_lag_2),
            'Petrol_Rolling_Std_7': roll_std_7
        }
        
        feat_df = pd.DataFrame([row_dict])[feature_cols]
        raw_ml_delta = float(best_model.predict(feat_df)[0])
        
        # Economic pass-through formula: In Indian petroleum economics, retail prices
        # react to international crude changes (~0.06 - 0.08 INR/L per $1 crude movement)
        economic_transmission = crude_step * 0.072
        
        if "Pause" in market_scenario and abs(crude_shock_pct) < 1.0:
            final_daily_delta = 0.0
        else:
            final_daily_delta = (raw_ml_delta * 0.20) + economic_transmission + drift_bias
            # If the crude step is negligible, allow OMC rate pauses
            if abs(crude_step) < 0.15 and "Volatility" in market_scenario:
                final_daily_delta = 0.0
        
        predicted_price = round(p_lag_1 + final_daily_delta, 2)
        
        # Expected price change relative to latest actual reference price
        change_in_inr = round(predicted_price - latest_actual_price, 2)
        pct_change = round((change_in_inr / latest_actual_price) * 100, 2)
        
        direction_icon = "▼" if change_in_inr < 0 else ("▲" if change_in_inr > 0 else "▬")
        
        forecast_records.append({
            'Day': f"Day {day}",
            'Forecast Date': next_date.strftime('%Y-%m-%d'),
            'Predicted Price (INR/L)': round(predicted_price, 2),
            'Expected Change (INR)': f"{direction_icon} {change_in_inr:+.2f} ₹" if change_in_inr != 0 else "▬ 0.00 ₹",
            'Percentage Change (%)': f"{pct_change:+.2f}%" if pct_change != 0 else "0.00%",
            'Direction': "DOWN" if change_in_inr < 0 else ("UP" if change_in_inr > 0 else "FLAT"),
            'Raw_Change': change_in_inr,
            'Daily_Delta': round(final_daily_delta, 2)
        })
        
        # Append simulated record to roll forward iteratively
        new_row = pd.DataFrame({
            'Date': [next_date],
            'Petrol_Price': [predicted_price],
            'Crude_Oil_USD': [current_crude],
            'USD_INR': [current_usd]
        })
        current_history = pd.concat([current_history, new_row], ignore_index=True)
        
    return pd.DataFrame(forecast_records), latest_actual_price, latest_date.strftime('%Y-%m-%d')

