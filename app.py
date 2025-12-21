import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tvDatafeed import TvDatafeed, Interval
import io
from datetime import datetime
import requests
import numpy as np

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="Stock DTE Meter", layout="wide")

# --- Initialize Session State ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'nifty_500_list' not in st.session_state:
    st.session_state.nifty_500_list = []

# --- Helper Functions ---

def get_nifty_500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        symbol_col = next((c for c in df.columns if c.lower() == 'symbol'), None)
        return df[symbol_col].str.upper().tolist() if symbol_col else []
    except Exception as e:
        return []

def calculate_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return round(100 - (100 / (1 + rs)).iloc[-1], 2)

def calculate_dte_metrics(df):
    try:
        if df is None or df.empty or len(df) < 15: return None
        current_price = df['close'].iloc[-1]
        
        # Calculate RSI
        rsi_val = calculate_rsi(df['close'])
        
        # Calculate DTE
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
        
        return {'gap': round(percent_gap, 2), 'price': round(current_price, 2), 'dte_lvl': round(dte_price, 2), 'rsi': rsi_val}
    except: return None

# Load Nifty 500 list into state once
if not st.session_state.nifty_500_list:
    st.session_state.nifty_500_list = get_nifty_500()

st.title("🎯 Stock DTE Meter")

# --- SECTION 1: QUICK LOOKUP ---
st.subheader("🔍 Single Stock Quick Lookup")
quick_sym = st.text_input("Enter Stock Symbol:").strip().upper()

if quick_sym:
    with st.spinner(f"Analyzing {quick_sym}..."):
        tv_quick = TvDatafeed()
        quick_results = []
        is_n500 = "Yes" if quick_sym in st.session_state.nifty_500_list else "No"
        
        for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
            hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tv_int, n_bars=100)
            data = calculate_dte_metrics(hist)
            if data:
                quick_results.append({
                    "Interval": label, "Price": data['price'], "DTE Lvl": data['dte_lvl'], "Gap%": data['gap'], "RSI": data['rsi']
                })
        
        if quick_results:
            st.write(f"**NIFTY 500 Member:** {is_n500}")
            st.table(pd.DataFrame(quick_results))
        else:
            st.warning("No data found.")

st.markdown("---")

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📂 Batch Scanner")
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])

stock_list = []
if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    symbol_col = next((c for c in df_input.columns if c.lower() == 'symbol'), None)
    if symbol_col:
        stock_list = df_input[symbol_col].dropna().astype(str).tolist()
else:
    if st.checkbox("Use NIFTY 500 List", value=False):
        stock_list = st.session_state.nifty_500_list

if stock_list:
    total_stocks = len(stock_list)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Start/Resume"): st.session_state.is_running = True
    with c2:
        if st.button("Pause"): st.session_state.is_running = False
    with c3:
        if st.button("Reset"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

    if st.session_state.is_running and st.session_state.last_index < total_stocks:
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / total_stocks)
        while st.session_state.last_index < total_stocks and st.session_state.is_running:
            i = st.session_state.last_index
            symbol = stock_list[i].strip().upper()
            try:
                metrics = {}
                is_n500 = "Yes" if symbol in st.session_state.nifty_500_list else "No"
                for label, tv_int in {'D': Interval.in_daily, 'W': Interval.in_weekly, 'M': Interval.in_monthly}.items():
                    hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        metrics[f'{label}_Gap%'] = data['gap']
                        metrics[f'{label}_RSI'] = data['rsi']
                        if label == 'D': curr_p = data['price']

                st.session_state.processed_results.append({
                    'Symbol': symbol, 'Nifty 500': is_n500, 'Price': curr_p,
                    'D_Gap%': metrics.get('D_Gap%', 0), 'D_RSI': metrics.get('D_RSI', 0),
                    'W_Gap%': metrics.get('W_Gap%', 0), 'W_RSI': metrics.get('W_RSI', 0),
                    'M_Gap%': metrics.get('M_Gap%', 0), 'M_RSI': metrics.get('M_RSI', 0)
                })
            except: pass
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / total_stocks)
            if st.session_state.last_index % 10 == 0: st.rerun()

    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        st.dataframe(df_res)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='DTE_RSI_Meter')
            workbook = writer.book
            worksheet = writer.sheets['DTE_RSI_Meter']
            worksheet.freeze_panes(1, 1)
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            for col_num, value in enumerate(df_res.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
        st.download_button("📥 Download Report", output.getvalue(), f"Stock_DTE_Meter_{datetime.now().strftime('%Y%m%d')}.xlsx")