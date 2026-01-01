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

# --- SHARED SESSION STATE ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'nifty_data_df' not in st.session_state:
    st.session_state.nifty_data_df = pd.DataFrame()

# --- SHARED HELPER FUNCTIONS ---

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
    except:
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

# Load NSE Data once
if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

# --- MODULE 1: DTE LOGIC (from app1.py) ---

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

# --- MODULE 2: MFI LOGIC (from app.py) ---

def get_mfi_metrics(df):
    if df is None or df.empty or len(df) < 2: return None
    df = df.copy()
    current_price = df['close'].iloc[-1]
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    
    # Bill Williams Coloring
    df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    def determine_color(row):
        if row['mfi_up'] and row['vol_up']: return "Green"
        if not row['mfi_up'] and not row['vol_up']: return "Fade"
        if row['mfi_up'] and not row['vol_up']: return "Fake"
        return "Squat"
    df['mfi_color'] = df.apply(determine_color, axis=1)
    
    max_mfi_idx = df['mfi'].idxmax()
    highest_mfi_val = df['mfi'].max()
    actual_price_at_peak = df.loc[max_mfi_idx, 'close']
    peak_color = df.loc[max_mfi_idx, 'mfi_color']
    
    # Calculate Tip Price
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
    tip_price = p_min + (rel_height * (p_max - p_min))
    
    prox_to_actual = abs((current_price - actual_price_at_peak) / actual_price_at_peak) * 100
    tip_to_actual_diff = abs((tip_price - actual_price_at_peak) / actual_price_at_peak) * 100
    is_filtered = (prox_to_actual < 2.0) and (tip_to_actual_diff > 15.0)
    
    return {
        'curr_price': round(current_price, 2),
        'actual_at_mfi': round(actual_price_at_peak, 2),
        'tip_price': round(tip_price, 2),
        'prox_pct': round(prox_to_actual, 2),
        'diff_pct': round(tip_to_actual_diff, 2),
        'mfi_date': max_mfi_idx.strftime('%Y-%m-%d %H:%M'),
        'mfi_color': peak_color,
        'passed_filter': is_filtered
    }

# --- NAVIGATION ---

st.sidebar.title("🛠️ Navigation")
page = st.sidebar.radio("Select Strategy:", ["DTE Meter (Volume-based)", "MFI Strategy Scanner"])

# When switching pages, we should offer the option to reset data
if st.sidebar.button("🗑️ Clear All Scan Results"):
    st.session_state.processed_results = []
    st.session_state.last_index = 0
    st.session_state.is_running = False
    st.rerun()

# --- PAGE 1: DTE METER ---
if page == "DTE Meter (Volume-based)":
    st.title("📊 Stock DTE Meter")
    st.markdown("Identifies price levels projected from the **highest volume peaks**.")
    
    # Quick Lookup
    st.subheader("🔍 Quick Lookup")
    quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
    if quick_sym:
        tv = TvDatafeed()
        q_res = []
        for lbl, tint in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
            hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
            data = calculate_dte_metrics(hist)
            if data: q_res.append({"Interval": lbl, "Price": data['price'], "DTE Price": data['dte_lvl'], "Gap%": data['gap']})
        if q_res:
            st.write(f"**Industry:** {get_industry_hybrid(quick_sym, st.session_state.nifty_data_df)}")
            st.table(pd.DataFrame(q_res))

    st.divider()
    
    # Batch Scanner (Logic adapted from app1.py)
    st.subheader("📑 Batch Scanner")
    uploaded_file = st.file_uploader("Upload Symbols Excel", type=["xlsx", "xls"], key="dte_upload")
    stock_list = []
    if uploaded_file:
        df_in = pd.read_excel(uploaded_file)
        sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
        if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
    elif st.checkbox("Use NIFTY 500 Index"):
        stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist()

    if stock_list:
        c1, c2, c3 = st.columns(3)
        if c1.button("🚀 Start Scanner"): st.session_state.is_running = True
        if c2.button("Pause"): st.session_state.is_running = False
        if c3.button("Reset"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

        if st.session_state.is_running:
            tv = TvDatafeed()
            progress_bar = st.progress(st.session_state.last_index / len(stock_list))
            while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
                sym = stock_list[st.session_state.last_index].strip().upper()
                industry = get_industry_hybrid(sym, st.session_state.nifty_data_df)
                try:
                    metrics = {}
                    cp = 0
                    for lbl, tint in {'D': Interval.in_daily, 'W': Interval.in_weekly}.items():
                        hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                        d = calculate_dte_metrics(hist)
                        if d:
                            metrics[f'{lbl}_Price'] = d['dte_lvl']
                            metrics[f'{lbl}_Gap%'] = d['gap']
                            cp = d['price']
                    st.session_state.processed_results.append({
                        'Symbol': sym, 'Industry': industry, 'Price': cp,
                        'D_DTE': metrics.get('D_Price', 0), 'D_Gap%': metrics.get('D_Gap%', 0),
                        'W_DTE': metrics.get('W_Price', 0), 'W_Gap%': metrics.get('W_Gap%', 0)
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
                df_res.to_excel(writer, index=False, sheet_name='DTE_Report')
            st.download_button("💾 Download Report", output.getvalue(), "DTE_Report.xlsx")

# --- PAGE 2: MFI STRATEGY SCANNER ---
elif page == "MFI Strategy Scanner":
    st.title("🎯 MFI High-Intensity Scanner")
    st.markdown("Filters stocks based on **Market Facilitation Index** intensity and price proximity.")
    
    timeframe = st.selectbox("Select Interval:", ["Hourly", "Daily", "Weekly"])
    interval_map = {"Hourly": Interval.in_1_hour, "Daily": Interval.in_daily, "Weekly": Interval.in_weekly}
    selected_interval = interval_map[timeframe]

    st.divider()
    
    uploaded_file = st.file_uploader("Upload Symbols Excel", type=["xlsx", "xls"], key="mfi_upload")
    stock_list = []
    if uploaded_file:
        df_in = pd.read_excel(uploaded_file)
        sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
        if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
    elif st.checkbox("Use NIFTY 500 Index"):
        stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist()

    if stock_list:
        c1, c2, c3 = st.columns(3)
        if c1.button("🚀 Run MFI Strategy"): st.session_state.is_running = True
        if c2.button("Pause"): st.session_state.is_running = False
        if c3.button("Reset"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

        if st.session_state.is_running:
            tv = TvDatafeed()
            progress_bar = st.progress(st.session_state.last_index / len(stock_list))
            while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
                sym = stock_list[st.session_state.last_index].strip().upper()
                try:
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=selected_interval, n_bars=100)
                    m = get_mfi_metrics(hist)
                    if m and m['passed_filter']:
                        st.session_state.processed_results.append({
                            'Symbol': sym, 'Sector': get_industry_hybrid(sym, st.session_state.nifty_data_df),
                            'Price': m['curr_price'], 'Actual @ Max MFI': m['actual_at_mfi'],
                            'Tip Price': m['tip_price'], 'MFI Color': m['mfi_color'],
                            'Prox%': m['prox_pct'], 'Intensity Diff%': m['diff_pct'], 'Peak Date': m['mfi_date']
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
                df_res.to_excel(writer, index=False, sheet_name='MFI_Report')
            st.download_button("💾 Download Report", output.getvalue(), "MFI_Report.xlsx")