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

st.set_page_config(page_title="Stock DTE & MFI Meter", layout="wide")

# --- Optimized Session State & Data Feed ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

@st.cache_resource
def get_tv_connection():
    """Initializes and caches the TradingView connection to prevent hanging."""
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

# --- Core Calculation Logic ---

def get_tip_price(df, target_series_name, target_idx):
    """Calculates the price equivalent for a specific bar's value (Vol or MFI)."""
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

def calculate_all_metrics(df):
    """Independent DTE and MFI metrics with dates, colors, and percentages."""
    if df is None or df.empty or len(df) < 15: return None
    
    df = df.copy()
    current_price = df['close'].iloc[-1]
    
    # 1. MFI Calculation & Bill Williams Color logic
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    
    def get_color(row):
        if row['mfi_up'] and row['vol_up']: return "Green"
        if not row['mfi_up'] and not row['vol_up']: return "Fade"
        if row['mfi_up'] and not row['vol_up']: return "Fake"
        return "Squat"
    df['mfi_color'] = df.apply(get_color, axis=1)

    # 2. Identify Peaks
    max_vol_idx = df['volume'].idxmax()
    max_mfi_idx = df['mfi'].idxmax()

    # 3. Independent Scaling for Vol and MFI
    vol_dte = get_tip_price(df, 'volume', max_vol_idx)
    mfi_tip = get_tip_price(df, 'mfi', max_mfi_idx)
    mfi_at_vol_peak = get_tip_price(df, 'mfi', max_vol_idx)

    return {
        'price': round(current_price, 2),
        'vol_dte': vol_dte,
        'vol_gap': round(((vol_dte - current_price) / current_price) * 100, 2),
        'vol_date': max_vol_idx.strftime('%Y-%m-%d'),
        'mfi_tip': mfi_tip,
        'mfi_gap': round(((mfi_tip - current_price) / current_price) * 100, 2),
        'mfi_date': max_mfi_idx.strftime('%Y-%m-%d'),
        'mfi_color': df.loc[max_mfi_idx, 'mfi_color'],
        'mfi_at_vol_peak': mfi_at_vol_peak
    }

# --- APP UI ---
master_data = fetch_nse_master_data()
tv = get_tv_connection()

st.title("📊 Complete Stock DTE & MFI Meter")

# Quick Lookup Section
quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
if quick_sym:
    q_res = []
    for lbl, inv in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
        hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=inv, n_bars=100)
        m = calculate_all_metrics(hist)
        if m:
            m.update({"Interval": lbl})
            q_res.append(m)
    if q_res:
        st.table(pd.DataFrame(q_res))

st.divider()

# Batch Scanner & Excel Upload
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
    if c1.button("🚀 Start/Resume"): st.session_state.is_running = True
    if c2.button("Pause"): st.session_state.is_running = False
    if c3.button("Reset Scanner"):
        st.session_state.processed_results = []
        st.session_state.last_index = 0
        st.session_state.is_running = False
        st.rerun()

    if st.session_state.processed_results:
        st.dataframe(pd.DataFrame(st.session_state.processed_results), use_container_width=True)

    if st.session_state.is_running and st.session_state.last_index < len(stock_list):
        progress = st.progress(st.session_state.last_index / len(stock_list))
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            sym = stock_list[st.session_state.last_index].strip().upper()
            row = {'Symbol': sym}
            for lbl, inv in {'D': Interval.in_daily, 'W': Interval.in_weekly}.items():
                hist = tv.get_hist(symbol=sym, exchange='NSE', interval=inv, n_bars=100)
                m = calculate_all_metrics(hist)
                if m:
                    row.update({
                        f'{lbl}_Price': m['price'],
                        f'{lbl}_Vol_DTE': m['vol_dte'], f'{lbl}_Vol_Gap%': m['vol_gap'], f'{lbl}_Vol_Date': m['vol_date'],
                        f'{lbl}_MFI_Tip': m['mfi_tip'], f'{lbl}_MFI_Gap%': m['mfi_gap'], f'{lbl}_MFI_Color': m['mfi_color'],
                        f'{lbl}_MFI_Date': m['mfi_date'], f'{lbl}_MFI@VolPeak': m['mfi_at_vol_peak']
                    })
            st.session_state.processed_results.append(row)
            st.session_state.last_index += 1
            if st.session_state.last_index % 5 == 0: st.rerun()