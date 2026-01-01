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

st.set_page_config(page_title="Stock DTE & MFI Meter", layout="wide")

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
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        return master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        return ticker.info.get('sector', 'N/A')
    except:
        return "N/A"

def calculate_mfi_details(df):
    """Calculates Bill Williams MFI and returns metrics for the highest MFI bar."""
    try:
        if df is None or df.empty or len(df) < 2: return None
        
        # 1. Basic MFI Calculation
        df = df.copy()
        df['mfi'] = (df['high'] - df['low']) / df['volume']
        
        # 2. MFI Coloring Logic
        df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
        df['vol_up'] = df['volume'] > df['volume'].shift(1)
        
        def get_color(row):
            if row['mfi_up'] and row['vol_up']: return "Green"
            if not row['mfi_up'] and not row['vol_up']: return "Fade"
            if row['mfi_up'] and not row['vol_up']: return "Fake"
            if not row['mfi_up'] and row['vol_up']: return "Squat"
            return "N/A"
        
        df['mfi_color'] = df.apply(get_color, axis=1)
        
        # 3. Find Highest MFI and its "Tip" price
        max_mfi_idx = df['mfi'].idxmax()
        highest_mfi_val = df['mfi'].max()
        highest_mfi_color = df.loc[max_mfi_idx, 'mfi_color']
        
        # Scale MFI to Price Axis (same logic as DTE Volume)
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
        
        current_price = df['close'].iloc[-1]
        mfi_gap = ((mfi_tip_price - current_price) / current_price) * 100
        
        return {
            'mfi_tip': round(mfi_tip_price, 2),
            'mfi_gap': round(mfi_gap, 2),
            'mfi_color': highest_mfi_color
        }
    except:
        return None

def calculate_dte_metrics(df):
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

# Startup
if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

st.title("📊 Stock DTE & MFI Meter")

# --- SECTION 1: QUICK LOOKUP ---
st.subheader("🔍 Single Stock Quick Lookup")
quick_sym = st.text_input("Enter NSE Symbol (e.g., RELIANCE):").strip().upper()

if quick_sym:
    industry = get_industry_hybrid(quick_sym, st.session_state.nifty_data_df)
    tv_quick = TvDatafeed()
    q_res = []
    mapping = {'Daily (1D)': Interval.in_daily, 'Weekly (1W)': Interval.in_weekly}
    
    with st.spinner(f"Fetching {quick_sym}..."):
        for lbl, tint in mapping.items():
            hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
            dte_data = calculate_dte_metrics(hist)
            mfi_data = calculate_mfi_details(hist)
            
            if dte_data and mfi_data:
                q_res.append({
                    "Interval": lbl, 
                    "Price": dte_data['price'], 
                    "DTE Price": dte_data['dte_lvl'], 
                    "DTE Gap%": dte_data['gap'],
                    "MFI Tip Price": mfi_data['mfi_tip'],
                    "MFI Gap%": mfi_data['mfi_gap'],
                    "MFI Color": mfi_data['mfi_color']
                })
    
    if q_res:
        st.table(pd.DataFrame(q_res))
    else:
        st.error("No data found for this symbol.")

st.divider()

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📑 Batch Scanner (DTE + MFI)")
uploaded_file = st.file_uploader("Upload Excel with 'Symbol' column", type=["xlsx", "xls"])

stock_list = []
if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
else:
    if st.checkbox("Use NIFTY 500 Index"): 
        stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist() if not st.session_state.nifty_data_df.empty else []

if stock_list:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Start/Resume"): st.session_state.is_running = True
    with col2:
        if st.button("Pause"): st.session_state.is_running = False
    with col3:
        if st.button("Reset Scanner"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        st.dataframe(df_res, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='DTE_MFI_Report')
        st.download_button("💾 Download Report", output.getvalue(), f"DTE_MFI_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")

    if st.session_state.is_running and st.session_state.last_index < len(stock_list):
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / len(stock_list))
        proc_map = {'D': Interval.in_daily, 'W': Interval.in_weekly}
        
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            idx = st.session_state.last_index
            sym = stock_list[idx].strip().upper()
            industry = get_industry_hybrid(sym, st.session_state.nifty_data_df)
            
            try:
                row_data = {'Symbol': sym, 'Industry': industry}
                for lbl, tint in proc_map.items():
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                    dte = calculate_dte_metrics(hist)
                    mfi = calculate_mfi_details(hist)
                    
                    if dte and mfi:
                        row_data['Price'] = dte['price']
                        row_data[f'{lbl}_DTE'] = dte['dte_lvl']
                        row_data[f'{lbl}_DTE_Gap%'] = dte['gap']
                        row_data[f'{lbl}_MFI_Tip'] = mfi['mfi_tip']
                        row_data[f'{lbl}_MFI_Gap%'] = mfi['mfi_gap']
                        row_data[f'{lbl}_MFI_Color'] = mfi['mfi_color']
                
                st.session_state.processed_results.append(row_data)
            except: pass
            
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 10 == 0: st.rerun()
        
        if st.session_state.last_index >= len(stock_list):
            st.session_state.is_running = False
            st.balloons()
            st.rerun()