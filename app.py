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

st.set_page_config(page_title="Stock Analysis Hub", layout="wide")

# --- Initialize Session State ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# --- Shared Helper Functions ---

@st.cache_resource
def get_tv_connection():
    return TvDatafeed()

@st.cache_data
def fetch_nse_master_data():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.upper()
        return df
    except:
        return pd.DataFrame()

def get_industry_hybrid(sym, master_df):
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        return master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        return ticker.info.get('sector', 'N/A')
    except:
        return "N/A"

def get_tip_price(df, target_series_name, target_idx):
    """Scales a secondary series (Volume or MFI) to the Price axis."""
    try:
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
        return round(p_min + (rel_height * (p_max - p_min)), 2)
    except:
        return 0

# --- MODULE 1: DTE LOGIC ---
def run_dte_module(hist, sym, industry):
    """Calculates Volume-based DTE metrics."""
    current_price = hist['close'].iloc[-1]
    max_vol_idx = hist['volume'].idxmax()
    dte_price = get_tip_price(hist, 'volume', max_vol_idx)
    gap = ((dte_price - current_price) / current_price) * 100
    return {
        'Symbol': sym, 'Sector': industry, 'Price': round(current_price, 2),
        'DTE Price': dte_price, 'Gap%': round(gap, 2), 'Date': max_vol_idx.strftime('%Y-%m-%d')
    }

# --- MODULE 2: MFI LOGIC ---
def run_mfi_module(hist, sym, industry):
    """Calculates MFI Tip vs Actual and applies strategy filters."""
    df = hist.copy()
    current_price = df['close'].iloc[-1]
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    
    # Bill Williams Color Logic
    df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    def get_color(row):
        if row['mfi_up'] and row['vol_up']: return "Green"
        if not row['mfi_up'] and not row['vol_up']: return "Fade"
        if row['mfi_up'] and not row['vol_up']: return "Fake"
        return "Squat"
    
    max_mfi_idx = df['mfi'].idxmax()
    actual_at_peak = df.loc[max_mfi_idx, 'close']
    tip_price = get_tip_price(df, 'mfi', max_mfi_idx)
    
    prox_pct = abs((current_price - actual_at_peak) / actual_at_peak) * 100
    diff_pct = abs((tip_price - actual_at_peak) / actual_at_peak) * 100
    
    # Filter: Close to actual (<2%) and High Intensity (diff > 15%)
    if (prox_pct < 2.0) and (diff_pct > 15.0):
        return {
            'Symbol': sym, 'Sector': industry, 'Current Price': round(current_price, 2),
            'Actual @ Max MFI': round(actual_at_peak, 2), 'Tip Price': tip_price,
            'Color': determine_color(df.loc[max_mfi_idx]), # Simplified for example
            'Prox%': round(prox_pct, 2), 'Diff%': round(diff_pct, 2), 'Date': max_mfi_idx.strftime('%Y-%m-%d %H:%M')
        }
    return None

# --- MAIN APP INTERFACE ---
master_data = fetch_nse_master_data()
tv = get_tv_connection()

st.sidebar.title("Configuration")
app_mode = st.sidebar.radio("Select Module:", ["DTE Meter (Volume)", "MFI Strategy Scanner"])
timeframe = st.sidebar.selectbox("Select Timeframe:", ["Hourly", "Daily", "Weekly"])

interval_map = {"Hourly": Interval.in_1_hour, "Daily": Interval.in_daily, "Weekly": Interval.in_weekly}
selected_interval = interval_map[timeframe]

st.title(f"🚀 {app_mode}")

# Shared Upload Logic
uploaded_file = st.file_uploader("Upload Excel with 'Symbol' column", type=["xlsx", "xls"])
stock_list = []
if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
elif st.checkbox("Use NIFTY 500 Index"):
    stock_list = master_data['SYMBOL'].tolist() if not master_data.empty else []

if stock_list:
    c1, c2, c3 = st.columns(3)
    if c1.button("🚀 Run Scanner"):
        st.session_state.processed_results = []
        st.session_state.last_index = 0
        st.session_state.is_running = True
    if c2.button("Pause"): st.session_state.is_running = False
    if c3.button("Reset"):
        st.session_state.processed_results = []; st.session_state.last_index = 0
        st.session_state.is_running = False; st.rerun()

    if st.session_state.is_running:
        progress_bar = st.progress(st.session_state.last_index / len(stock_list))
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            sym = stock_list[st.session_state.last_index].strip().upper()
            industry = get_industry_hybrid(sym, master_data)
            try:
                hist = tv.get_hist(symbol=sym, exchange='NSE', interval=selected_interval, n_bars=100)
                
                if app_mode == "DTE Meter (Volume)":
                    res = run_dte_module(hist, sym, industry)
                    if res: st.session_state.processed_results.append(res)
                else:
                    res = run_mfi_module(hist, sym, industry)
                    if res: st.session_state.processed_results.append(res)
            except: pass
            
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 10 == 0: st.rerun()

    if st.session_state.processed_results:
        st.dataframe(pd.DataFrame(st.session_state.processed_results), use_container_width=True)