import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tvDatafeed import TvDatafeed, Interval
import yfinance as yf
import io
from datetime import datetime
import requests

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="Mod Stock DTE & MFI Meter", layout="wide")

# --- CORE CALCULATION LOGIC ---

def get_tip_price(df, target_series_name, target_idx):
    """Calculates the price equivalent for a specific bar's value in a secondary series (Vol or MFI)."""
    try:
        # Scale the secondary series (Vol/MFI) to the Price axis to find the 'tip'
        fig, ax1 = plt.subplots()
        ax1.plot(df.index, df['close'])
        ax2 = ax1.twinx()
        ax2.bar(df.index, df[target_series_name])
        
        p_min, p_max = ax1.get_ylim()
        s_min, s_max = ax2.get_ylim()
        plt.close(fig)
        
        val = df.loc[target_idx, target_series_name]
        s_range = (s_max - s_min)
        if s_range == 0: return 0
        
        rel_height = (val - s_min) / s_range
        tip_price = p_min + (rel_height * (p_max - p_min))
        return round(tip_price, 2)
    except:
        return 0

def calculate_combined_metrics(df):
    """Independent DTE and MFI metrics with peak dates and gap percentages."""
    if df is None or df.empty or len(df) < 15: return None
    
    df = df.copy()
    current_price = df['close'].iloc[-1]
    
    # 1. MFI Calculation & Coloring
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    
    def get_color(row):
        if row['mfi_up'] and row['vol_up']: return "Green"
        if not row['mfi_up'] and not row['vol_up']: return "Fade"
        if row['mfi_up'] and not row['vol_up']: return "Fake"
        return "Squat"
    
    df['mfi_color'] = df.apply(get_color, axis=1)

    # 2. Identify Peak Indices and Dates
    max_vol_idx = df['volume'].idxmax()
    max_mfi_idx = df['mfi'].idxmax()

    # 3. Calculate Independent Tip Prices & Percentages
    vol_dte_price = get_tip_price(df, 'volume', max_vol_idx)
    mfi_dte_price = get_tip_price(df, 'mfi', max_mfi_idx)
    
    # MFI tip price at the specific bar where Volume peaked
    mfi_at_vol_peak_price = get_tip_price(df, 'mfi', max_vol_idx)

    return {
        'price': round(current_price, 2),
        # Volume DTE Details
        'vol_dte': vol_dte_price,
        'vol_gap_pct': round(((vol_dte_price - current_price) / current_price) * 100, 2),
        'vol_peak_date': max_vol_idx.strftime('%Y-%m-%d'),
        # MFI DTE Details
        'mfi_dte': mfi_dte_price,
        'mfi_gap_pct': round(((mfi_dte_price - current_price) / current_price) * 100, 2),
        'mfi_peak_date': max_mfi_idx.strftime('%Y-%m-%d'),
        'mfi_peak_color': df.loc[max_mfi_idx, 'mfi_color'],
        # Specific Request: MFI Height at Volume Peak
        'mfi_at_vol_peak': mfi_at_vol_peak_price
    }

# --- APP UI ---

st.title("📊 Enhanced Stock DTE & MFI Meter")

# Quick Lookup Section
quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
if quick_sym:
    tv = TvDatafeed()
    intervals = {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}
    results = []
    
    for lbl, inv in intervals.items():
        hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=inv, n_bars=100)
        m = calculate_combined_metrics(hist)
        if m:
            results.append({
                "Interval": lbl, 
                "Price": m['price'],
                "Vol DTE": m['vol_dte'], 
                "Vol Gap %": m['vol_gap_pct'], 
                "Vol Peak Date": m['vol_peak_date'],
                "MFI Tip": m['mfi_dte'], 
                "MFI Gap %": m['mfi_gap_pct'], 
                "MFI Color": m['mfi_peak_color'], 
                "MFI Peak Date": m['mfi_peak_date'],
                "MFI @ Vol Peak": m['mfi_at_vol_peak']
            })
    if results:
        st.table(pd.DataFrame(results))

st.divider()

# Batch Scanner functionality can call the same 'calculate_combined_metrics' function to populate the report.