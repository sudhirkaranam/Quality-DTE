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

# --- INITIALIZE SESSION STATE ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'nifty_data_df' not in st.session_state:
    st.session_state.nifty_data_df = pd.DataFrame()
if 'current_module' not in st.session_state:
    st.session_state.current_module = "Scanner"

# --- SHARED HELPER FUNCTIONS ---

@st.cache_resource
def get_tv_connection():
    return TvDatafeed()

@st.cache_data
def fetch_nse_master_data():
    """Fetches the official Nifty 500 list."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.upper()
        return df
    except:
        return pd.DataFrame()

def get_stock_info(sym, master_df):
    """Retrieves Industry and Nifty 500 status."""
    is_nifty500 = "No"
    industry = "N/A"
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        is_nifty500 = "Yes"
        industry = master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    else:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            industry = ticker.info.get('sector', 'N/A')
        except: pass
    return industry, is_nifty500

def get_tip_price(df, target_series_name, target_idx):
    """Scales Volume or MFI peaks to the Price axis."""
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
    except: return 0

# --- MODULE CALCULATIONS ---

def calculate_dte_metrics(df):
    """Volume-based DTE logic."""
    try:
        if df is None or df.empty or len(df) < 15: return None
        current_price = df['close'].iloc[-1]
        max_vol_idx = df['volume'].idxmax()
        dte_price = get_tip_price(df, 'volume', max_vol_idx)
        percent_gap = ((dte_price - current_price) / current_price) * 100
        return {
            'Price': round(current_price, 2), 
            'DTE_Price': dte_price, 
            'Gap%': round(percent_gap, 2), 
            'Peak_Date': max_vol_idx.strftime('%Y-%m-%d')
        }
    except: return None

def get_mfi_metrics(df):
    """Calculates MFI Tip and check logic."""
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
    actual_at_peak = df.loc[max_mfi_idx, 'close']
    tip_price = get_tip_price(df, 'mfi', max_mfi_idx)
    
    # Check if price reached Tip post-peak
    pos = df.index.get_loc(max_mfi_idx)
    post_peak_df = df.iloc[pos+1:]
    reached_tip = "No"
    if not post_peak_df.empty:
        if post_peak_df['close'].max() >= tip_price:
            reached_tip = "Yes"

    prox_to_actual = abs((current_price - actual_at_peak) / actual_at_peak) * 100
    tip_to_actual_diff = abs((tip_price - actual_at_peak) / actual_at_peak) * 100
    is_filtered = (prox_to_actual < 2.0) and (tip_to_actual_diff > 15.0)
    
    return {
        'Price': round(current_price, 2), 'Actual_Peak': round(actual_at_peak, 2),
        'Tip_Price': round(tip_price, 2), 'Reached_Tip': reached_tip,
        'Prox%': round(prox_to_actual, 2), 'Intensity%': round(tip_to_actual_diff, 2),
        'Peak_Date': max_mfi_idx.strftime('%Y-%m-%d %H:%M'), 'Color': df.loc[max_mfi_idx, 'mfi_color'],
        'passed': is_filtered
    }

# --- NAVIGATION AREA ---

if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

st.sidebar.title("🛠️ Navigation")
page = st.sidebar.radio("Navigation", ["Scanner", "DTE Meter"])

if page != st.session_state.current_module:
    st.session_state.processed_results = []
    st.session_state.last_index = 0
    st.session_state.is_running = False
    st.session_state.current_module = page
    st.rerun()

# --- MAIN UI ---

if page == "Scanner":
    st.title("🎯 MFI Multi-Timeframe Scanner")
    st.subheader("🔍 Single Stock NSE Lookup")
    quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
    if quick_sym:
        industry, is_n500 = get_stock_info(quick_sym, st.session_state.nifty_data_df)
        tv = get_tv_connection()
        q_rows = []
        for lbl, tint in {'Hourly': Interval.in_1_hour, 'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
            hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
            m = get_mfi_metrics(hist)
            if m:
                m.update({'TF': lbl})
                q_rows.append(m)
        if q_rows:
            st.write(f"**Sector:** {industry} | **Nifty 500:** {is_n500}")
            st.table(pd.DataFrame(q_rows).drop(columns=['passed']))

elif page == "DTE Meter":
    st.title("📊 Stock DTE Meter")
    st.subheader("🔍 Quick DTE Lookup")
    quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
    if quick_sym:
        industry, is_n500 = get_stock_info(quick_sym, st.session_state.nifty_data_df)
        tv = get_tv_connection()
        q_res = []
        for lbl, tint in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
            hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
            d = calculate_dte_metrics(hist)
            if d:
                d.update({"TF": lbl, "Sector": industry, "Nifty 500": is_n500})
                q_res.append(d)
        if q_res: st.table(pd.DataFrame(q_res))

st.divider()

# --- BATCH SCANNER ---

st.subheader("📑 Batch Scanner")
uploaded_file = st.file_uploader("Upload Symbols Excel", type=["xlsx", "xls"])
stock_list = []
source_mode = "manual" # Default

if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
    source_mode = "upload"
elif st.checkbox("Use NIFTY 500 Index"):
    stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist()
    source_mode = "nifty500"

if stock_list:
    c1, c2, c3 = st.columns(3)
    if c1.button("🚀 Start Scanner"): 
        st.session_state.processed_results = []; st.session_state.last_index = 0
        st.session_state.is_running = True
    if c2.button("Pause"): st.session_state.is_running = False
    if c3.button("Reset"):
        st.session_state.processed_results = []; st.session_state.last_index = 0
        st.session_state.is_running = False; st.rerun()

    progress_placeholder = st.empty()
    table_placeholder = st.empty()

    if st.session_state.processed_results:
        df_display = pd.DataFrame(st.session_state.processed_results)
        table_placeholder.dataframe(df_display, width=1200)
        
        dl_date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{page}_Report_{dl_date}.xlsx"
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_display.to_excel(writer, index=False, sheet_name='Report')
        st.download_button("💾 Download Results", output.getvalue(), filename)

    if st.session_state.is_running and st.session_state.last_index < len(stock_list):
        tv = get_tv_connection()
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            sym = stock_list[st.session_state.last_index].strip().upper()
            industry, is_n500 = get_stock_info(sym, st.session_state.nifty_data_df)
            try:
                if page == "DTE Meter":
                    # For Upload: Show all rows | For Nifty500: Logic remains multi-row
                    for lbl, tint in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
                        hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                        d = calculate_dte_metrics(hist)
                        if d:
                            st.session_state.processed_results.append({
                                'Symbol': sym, 'TF': lbl, 'Nifty 500': is_n500, 'Sector': industry,
                                'Price': d['Price'], 'DTE_Price': d['DTE_Price'], 
                                'Gap%': d['Gap%'], 'Date': d['Peak_Date']
                            })
                else: # MFI Mode
                    for lbl, tint in {'Hourly': Interval.in_1_hour, 'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
                        hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                        m = get_mfi_metrics(hist)
                        if m:
                            # CONDITIONAL LOGIC
                            # Nifty500: Filter symbols | Upload: Show all details
                            if source_mode == "upload" or m['passed']:
                                st.session_state.processed_results.append({
                                    'Symbol': sym, 'TF': lbl, 'Nifty 500': is_n500, 'Sector': industry, 
                                    'Price': m['Price'], 'Actual_Pk': m['Actual_Peak'], 'Tip': m['Tip_Price'],
                                    'Reach?': m['Reached_Tip'], 'Color': m['Color'], 'Int%': m['Intensity%'], 'Date': m['Peak_Date']
                                })
            except: pass
            st.session_state.last_index += 1
            progress_placeholder.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 5 == 0: st.rerun()