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
if 'nifty_data_df' not in st.session_state:
    st.session_state.nifty_data_df = pd.DataFrame()

# --- Shared Helper Functions ---

@st.cache_data
def fetch_nse_master_data():
    """Fetches the official Nifty 500 list with Industry classification."""
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
    """Priority 1: NSE Master List | Priority 2: Yahoo Finance."""
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        return master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        return ticker.info.get('sector', 'N/A')
    except:
        return "N/A"

def get_tip_price(df, target_series_name, target_idx):
    """Generic helper to scale Volume or MFI to the Price axis."""
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

# --- MODULE 1: ORIGINAL DTE LOGIC (AS IS) ---

def calculate_dte_metrics(df):
    """Original DTE calculation logic from user file."""
    try:
        if df is None or df.empty or len(df) < 15: return None
        current_price = df['close'].iloc[-1]
        
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
        percent_gap = ((dte_price - current_price) / current_price) * 100
        
        return {'gap': round(percent_gap, 2), 'price': round(current_price, 2), 'dte_lvl': round(dte_price, 2)}
    except: return None

# --- MODULE 2: NEW MFI LOGIC ---

def calculate_mfi_strategy(df):
    """MFI intensity logic with 2% proximity and 15% tip-difference filters."""
    try:
        if df is None or df.empty or len(df) < 15: return None
        df = df.copy()
        current_price = df['close'].iloc[-1]
        
        # MFI Calculation
        df['mfi'] = (df['high'] - df['low']) / df['volume']
        
        # Color Logic
        df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
        df['vol_up'] = df['volume'] > df['volume'].shift(1)
        def get_color(row):
            if row['mfi_up'] and row['vol_up']: return "Green"
            if not row['mfi_up'] and not row['vol_up']: return "Fade"
            if row['mfi_up'] and not row['vol_up']: return "Fake"
            return "Squat"
        df['mfi_color'] = df.apply(get_color, axis=1)

        max_mfi_idx = df['mfi'].idxmax()
        actual_at_peak = df.loc[max_mfi_idx, 'close']
        tip_price = get_tip_price(df, 'mfi', max_mfi_idx)
        
        prox_pct = abs((current_price - actual_at_peak) / actual_at_peak) * 100
        diff_pct = abs((tip_price - actual_at_peak) / actual_at_peak) * 100
        
        # Filter Logic
        if (prox_pct < 2.0) and (diff_pct > 15.0):
            return {
                'curr_price': round(current_price, 2),
                'actual_at_mfi': round(actual_at_peak, 2),
                'tip_price': tip_price,
                'mfi_color': df.loc[max_mfi_idx, 'mfi_color'],
                'prox_pct': round(prox_pct, 2),
                'diff_pct': round(diff_pct, 2),
                'date': max_mfi_idx.strftime('%Y-%m-%d %H:%M')
            }
        return None
    except: return None

# --- MAIN INTERFACE ---

if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

st.sidebar.title("Module Selection")
app_mode = st.sidebar.radio("Select Analysis Module:", ["DTE Meter (Volume)", "MFI Strategy Scanner"])
timeframe = st.sidebar.selectbox("Select Timeframe:", ["Hourly", "Daily", "Weekly"])

interval_map = {"Hourly": Interval.in_1_hour, "Daily": Interval.in_daily, "Weekly": Interval.in_weekly}
selected_interval = interval_map[timeframe]

st.title(f"🚀 {app_mode}")

# Unified Batch Scanner Logic
uploaded_file = st.file_uploader("Upload Excel with 'Symbol' column", type=["xlsx", "xls"])
stock_list = []
if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
elif st.checkbox("Use NIFTY 500 Index"):
    stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist()

if stock_list:
    c1, c2, c3 = st.columns(3)
    if c1.button("🚀 Run Scanner"):
        st.session_state.processed_results = []
        st.session_state.last_index = 0
        st.session_state.is_running = True
    if c2.button("Pause"): st.session_state.is_running = False
    if c3.button("Reset Scanner"):
        st.session_state.processed_results = []; st.session_state.last_index = 0
        st.session_state.is_running = False; st.rerun()

    if st.session_state.is_running:
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / len(stock_list))
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            sym = stock_list[st.session_state.last_index].strip().upper()
            industry = get_industry_hybrid(sym, st.session_state.nifty_data_df)
            try:
                hist = tv.get_hist(symbol=sym, exchange='NSE', interval=selected_interval, n_bars=100)
                
                if app_mode == "DTE Meter (Volume)":
                    res = calculate_dte_metrics(hist)
                    if res:
                        st.session_state.processed_results.append({
                            'Symbol': sym, 'Sector': industry, 'Price': res['price'],
                            'DTE Price': res['dte_lvl'], 'Gap%': res['gap']
                        })
                else:
                    res = calculate_mfi_strategy(hist)
                    if res:
                        st.session_state.processed_results.append({
                            'Symbol': sym, 'Sector': industry, 'Current Price': res['curr_price'],
                            'Actual @ Max MFI': res['actual_at_mfi'], 'Tip Price': res['tip_price'],
                            'Color': res['mfi_color'], 'Prox%': res['prox_pct'], 'Diff%': res['diff_pct'], 'Peak Date': res['date']
                        })
            except: pass
            
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 10 == 0: st.rerun()

    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        st.dataframe(df_res, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='Report')
        st.download_button("💾 Download Full Report", output.getvalue(), "Stock_Analysis_Report.xlsx")