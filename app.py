import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tvDatafeed import TvDatafeed, Interval
import yfinance as yf
import io
import time
from datetime import datetime
import requests

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="Mod Stock DTE & MFI Meter", layout="wide")

# --- Initialize Session State ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'nifty_data_df' not in st.session_state:
    st.session_state.nifty_data_df = pd.DataFrame()

# --- Helper Functions ---

def fetch_nse_master_data():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.upper()
        return df
    except Exception:
        return pd.DataFrame()

def get_industry_hybrid(sym, master_df):
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        return master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        return ticker.info.get('sector', 'N/A')
    except:
        return "N/A"

def calculate_mfi_details(df):
    """Calculates Market Facilitation Index (MFI) metrics and peak details."""
    try:
        if df is None or df.empty or len(df) < 2: return None
        
        df = df.copy()
        # Bill Williams MFI: (High - Low) / Volume
        df['mfi'] = (df['high'] - df['low']) / df['volume']
        
        # MFI Coloring Logic
        df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
        df['vol_up'] = df['volume'] > df['volume'].shift(1)
        
        def get_color(row):
            if row['mfi_up'] and row['vol_up']: return "Green"
            if not row['mfi_up'] and not row['vol_up']: return "Fade"
            if row['mfi_up'] and not row['vol_up']: return "Fake"
            if not row['mfi_up'] and row['vol_up']: return "Squat"
            return "N/A"
        
        df['mfi_color'] = df.apply(get_color, axis=1)
        
        max_mfi_idx = df['mfi'].idxmax()
        highest_mfi_val = df['mfi'].max()
        peak_date = max_mfi_idx.strftime('%Y-%m-%d')
        
        # Scale MFI value to price scale
        fig, ax1 = plt.subplots()
        ax1.plot(df.index, df['close'])
        ax2 = ax1.twinx()
        ax2.bar(df.index, df['mfi'])
        p_min, p_max = ax1.get_ylim()
        m_min, m_max = ax2.get_ylim()
        plt.close(fig)
        
        m_range = (m_max - m_min)
        if m_range == 0: return None
        rel_height = (highest_mfi_val - m_min) / m_range
        mfi_tip_price = p_min + (rel_height * (p_max - p_min))
        
        return {
            'mfi_tip': round(mfi_tip_price, 2),
            'mfi_color': df.loc[max_mfi_idx, 'mfi_color'],
            'mfi_date': peak_date
        }
    except:
        return None

def calculate_dte_metrics(df):
    """Calculates Volume-based DTE metrics and peak details."""
    try:
        if df is None or df.empty or len(df) < 15: return None
        current_price = df['close'].iloc[-1]
        
        max_vol_idx = df['volume'].idxmax()
        peak_date = max_vol_idx.strftime('%Y-%m-%d')
        
        fig, ax1 = plt.subplots()
        ax1.plot(df.index, df['close'])
        ax2 = ax1.twinx()
        ax2.bar(df.index, df['volume'])
        p_min, p_max = ax1.get_ylim()
        v_min, v_max = ax2.get_ylim()
        plt.close(fig) 
        
        max_vol = df['volume'].max()
        v_range = (v_max - v_min)
        if v_range == 0: return None
        rel_height = (max_vol - v_min) / v_range
        dte_price = p_min + (rel_height * (p_max - p_min))
        
        return {
            'price': round(current_price, 2), 
            'dte_lvl': round(dte_price, 2), 
            'dte_date': peak_date
        }
    except: return None

if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

st.title("📊 Stock DTE & MFI Meter")

# --- SECTION 1: QUICK LOOKUP ---
st.subheader("🔍 Single Stock Quick Lookup")
quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()

if quick_sym:
    tv_quick = TvDatafeed()
    q_res = []
    mapping = {'Daily (1D)': Interval.in_daily, 'Weekly (1W)': Interval.in_weekly}
    
    for lbl, tint in mapping.items():
        hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
        dte = calculate_dte_metrics(hist)
        mfi = calculate_mfi_details(hist)
        if dte and mfi:
            q_res.append({
                "Interval": lbl, "Price": dte['price'], 
                "DTE Price": dte['dte_lvl'], "DTE Date": dte['dte_date'],
                "MFI Tip": mfi['mfi_tip'], "MFI Color": mfi['mfi_color'], "MFI Date": mfi['mfi_date']
            })
    if q_res: st.table(pd.DataFrame(q_res))

st.divider()

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📑 Batch Scanner")
# (Upload logic same as previous version)
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
stock_list = []
if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()

if stock_list:
    if st.button("🚀 Start Scanner"): st.session_state.is_running = True
    
    if st.session_state.is_running:
        tv = TvDatafeed()
        for idx in range(st.session_state.last_index, len(stock_list)):
            sym = stock_list[idx].strip().upper()
            row = {'Symbol': sym}
            for lbl, tint in {'D': Interval.in_daily, 'W': Interval.in_weekly}.items():
                hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                d = calculate_dte_metrics(hist)
                m = calculate_mfi_details(hist)
                if d and m:
                    row.update({
                        f'{lbl}_DTE': d['dte_lvl'], f'{lbl}_DTE_Date': d['dte_date'],
                        f'{lbl}_MFI_Tip': m['mfi_tip'], f'{lbl}_MFI_Color': m['mfi_color'], f'{lbl}_MFI_Date': m['mfi_date']
                    })
            st.session_state.processed_results.append(row)
            st.session_state.last_index += 1
            if idx % 5 == 0: st.rerun()
        st.dataframe(pd.DataFrame(st.session_state.processed_results))