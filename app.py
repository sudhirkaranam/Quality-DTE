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

st.set_page_config(page_title="MFI Strategy Scanner", layout="wide")

# --- Optimized Session State ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

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

def get_mfi_metrics(df):
    """Calculates MFI Tip vs Actual Price, MFI Color, and applies filters."""
    if df is None or df.empty or len(df) < 2: return None
    
    df = df.copy()
    current_price = df['close'].iloc[-1]
    
    # 1. MFI Calculation: (High - Low) / Volume
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    
    # 2. MFI Coloring Logic (Bill Williams)
    df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    
    def determine_color(row):
        if row['mfi_up'] and row['vol_up']: return "Green"
        if not row['mfi_up'] and not row['vol_up']: return "Fade"
        if row['mfi_up'] and not row['vol_up']: return "Fake"
        return "Squat"
    
    df['mfi_color'] = df.apply(determine_color, axis=1)
    
    # 3. Identify Peak MFI
    max_mfi_idx = df['mfi'].idxmax()
    highest_mfi_val = df['mfi'].max()
    actual_price_at_peak = df.loc[max_mfi_idx, 'close']
    peak_color = df.loc[max_mfi_idx, 'mfi_color']
    
    # 4. Calculate "Price at Tip" via scaling
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
    
    # 5. Filter Logic:
    # - Current Price close to (<2%) Actual Price at highest MFI
    # - Difference between Actual and Tip Price > 15%
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

# --- APP UI ---
master_data = fetch_nse_master_data()
tv = get_tv_connection()

st.title("🎯 MFI High-Intensity Scanner")

# Interval Selection
timeframe = st.selectbox("Select Timeframe Interval:", ["Hourly", "Daily", "Weekly"])
interval_map = {
    "Hourly": Interval.in_1_hour,
    "Daily": Interval.in_daily,
    "Weekly": Interval.in_weekly
}
selected_interval = interval_map[timeframe]

st.divider()

# Batch Scanner Section
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
    if c1.button("🚀 Run Strategy Scanner"): 
        st.session_state.processed_results = []
        st.session_state.last_index = 0
        st.session_state.is_running = True
    if c2.button("Pause"): st.session_state.is_running = False
    if c3.button("Clear Results"):
        st.session_state.processed_results = []
        st.session_state.last_index = 0
        st.session_state.is_running = False
        st.rerun()

    if st.session_state.is_running:
        progress_bar = st.progress(st.session_state.last_index / len(stock_list))
        
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            sym = stock_list[st.session_state.last_index].strip().upper()
            try:
                hist = tv.get_hist(symbol=sym, exchange='NSE', interval=selected_interval, n_bars=100)
                m = get_mfi_metrics(hist)
                
                if m and m['passed_filter']:
                    row = {
                        'Symbol': sym,
                        'Sector': get_industry_hybrid(sym, master_data),
                        'Current Price': m['curr_price'],
                        'Actual @ Max MFI': m['actual_at_mfi'],
                        'Tip Price @ Max MFI': m['tip_price'],
                        'MFI Color': m['mfi_color'],
                        'Prox to Actual %': m['prox_pct'],
                        'Tip/Actual Diff %': m['diff_pct'],
                        'Peak Date': m['mfi_date']
                    }
                    st.session_state.processed_results.append(row)
            except:
                pass
            
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 10 == 0: st.rerun()

    if st.session_state.processed_results:
        st.subheader(f"📈 Filtered Results ({timeframe})")
        df_display = pd.DataFrame(st.session_state.processed_results)
        st.dataframe(df_display, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='MFI_Report')
        st.download_button("💾 Download MFI Report", output.getvalue(), f"MFI_{timeframe}_Report.xlsx")