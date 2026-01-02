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

# --- 1. HELPER FUNCTIONS ---

@st.cache_resource
def get_tv_connection():
    return TvDatafeed()

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

def get_stock_info(sym, master_df):
    industry = "N/A"
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        industry = master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    else:
        try:
            ticker = yf.Ticker(f"{sym}.NS")
            industry = ticker.info.get('sector', 'N/A')
        except: pass
    return industry

def get_tip_price(df, target_series_name, target_idx):
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

def get_mfi_metrics(df):
    if df is None or df.empty or len(df) < 2: return None
    df = df.copy()
    current_price = df['close'].iloc[-1]
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    max_mfi_idx = df['mfi'].idxmax()
    actual_at_peak = df.loc[max_mfi_idx, 'close']
    tip_price = get_tip_price(df, 'mfi', max_mfi_idx)
    
    # Check if price reached Tip post-peak
    post_peak_df = df.iloc[df.index.get_loc(max_mfi_idx)+1:]
    reached = "Yes" if not post_peak_df.empty and post_peak_df['close'].max() >= tip_price else "No"
    
    # Updated Intensity Formula: Current Price vs Tip Price
    prox_pct = abs((current_price - actual_at_peak) / actual_at_peak) * 100
    int_pct = abs((tip_price - current_price) / current_price) * 100
    
    return {
        'Price': round(current_price, 2), 'Actual_Peak': round(actual_at_peak, 2),
        'Tip': round(tip_price, 2), 'Reached': reached,
        'Prox%': round(prox_pct, 2), 'Int%': round(int_pct, 2),
        'Date': max_mfi_idx.strftime('%Y-%m-%d %H:%M'),
        'passed': (prox_pct < 2.0) and (int_pct > 15.0)
    }

def calculate_dte_metrics(df):
    try:
        if df is None or df.empty or len(df) < 15: return None
        current_price = df['close'].iloc[-1]
        max_vol_idx = df['volume'].idxmax()
        dte_price = get_tip_price(df, 'volume', max_vol_idx)
        post_peak_df = df.iloc[df.index.get_loc(max_vol_idx)+1:]
        reached = "Yes" if not post_peak_df.empty and post_peak_df['close'].max() >= dte_price else "No"
        # Intensity logic for DTE: Current Price vs DTE Price
        percent_gap = abs((dte_price - current_price) / current_price) * 100
        return {
            'Price': round(current_price, 2), 
            'DTE_Price': dte_price, 
            'Reached': reached, 
            'Int%': round(percent_gap, 2), 
            'Peak_Date': max_vol_idx.strftime('%Y-%m-%d')
        }
    except: return None

# --- 2. INITIALIZE SESSION STATE ---

if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'nifty_data_df' not in st.session_state:
    st.session_state.nifty_data_df = fetch_nse_master_data()
if 'current_module' not in st.session_state:
    st.session_state.current_module = "Scanner"

# --- 3. NAVIGATION ---

st.sidebar.title("🛠️ Navigation")
page = st.sidebar.radio("Modules :", ["Scanner", "DTE Meter"])

if page != st.session_state.current_module:
    st.session_state.processed_results = []
    st.session_state.last_index = 0
    st.session_state.is_running = False
    st.session_state.current_module = page
    st.rerun()

# --- 4. MAIN UI ---

st.title(f"🚀 {page} Module")

if page == "Scanner":
    st.subheader("🔍 Single Stock NSE Lookup")
    quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
    if quick_sym:
        industry = get_stock_info(quick_sym, st.session_state.nifty_data_df)
        tv = get_tv_connection()
        q_rows = []
        for lbl, tint in {'Hourly': Interval.in_1_hour, 'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
            hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
            m = get_mfi_metrics(hist)
            if m:
                m.update({'TF': lbl}); q_rows.append(m)
        if q_rows:
            st.write(f"**Sector:** {industry}")
            st.table(pd.DataFrame(q_rows).drop(columns=['passed']))

elif page == "DTE Meter":
    st.subheader("🔍 Quick DTE Lookup")
    quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()
    if quick_sym:
        industry = get_stock_info(quick_sym, st.session_state.nifty_data_df)
        tv = get_tv_connection()
        q_res = []
        for lbl, tint in {'Hourly': Interval.in_1_hour, 'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
            hist = tv.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
            d = calculate_dte_metrics(hist)
            if d:
                d.update({"TF": lbl}); q_res.append(d)
        if q_res:
            st.write(f"**Sector:** {industry}")
            st.table(pd.DataFrame(q_res))

# --- 5. BATCH SCANNER UI ---

st.divider()
st.subheader("📑 Batch Scanner")
uploaded_file = st.file_uploader("Upload Symbols Excel", type=["xlsx", "xls"])
stock_list = []
source_mode = "manual"
selected_tf_label = "All"

if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
    source_mode = "upload"
elif st.checkbox("Use NIFTY 500 Index"):
    stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist()
    source_mode = "nifty500"
    selected_tf_label = st.selectbox("Filter NIFTY 500 by Timeframe:", ["All", "Hourly", "Daily", "Weekly"])

c1, c2, c3 = st.columns(3)
if c1.button("▶️ Start Scan"):
    st.session_state.processed_results = []; st.session_state.last_index = 0
    st.session_state.is_running = True
if c2.button("⏸️ Pause"): st.session_state.is_running = False
if c3.button("🔄 Reset"):
    st.session_state.processed_results = []; st.session_state.last_index = 0
    st.session_state.is_running = False; st.rerun()

# --- 6. RESULTS DISPLAY ---

if st.session_state.processed_results:
    st.write("---")
    df_res = pd.DataFrame(st.session_state.processed_results)
    
    if source_mode == "nifty500" and selected_tf_label == "All" and not df_res.empty:
        df_res = df_res.drop_duplicates(subset=['Symbol'])
        
    st.dataframe(df_res, width=1400)
    dl_date = datetime.now().strftime("%Y-%m-%d")
    filename = f"{page}_{selected_tf_label}_Report_{dl_date}.xlsx"
    
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False)
    st.download_button(f"💾 Download {selected_tf_label} Report", output.getvalue(), filename)

# --- 7. PROCESSING ENGINE ---

if st.session_state.is_running and st.session_state.last_index < len(stock_list):
    progress_bar = st.progress(st.session_state.last_index / len(stock_list))
    tv = get_tv_connection()
    chunk = stock_list[st.session_state.last_index : st.session_state.last_index + 5]
    
    tf_map = {'Hourly': Interval.in_1_hour, 'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}
    if selected_tf_label != "All":
        active_tfs = {selected_tf_label: tf_map[selected_tf_label]}
    else:
        active_tfs = {'H': Interval.in_1_hour, 'D': Interval.in_daily, 'W': Interval.in_weekly}

    for sym in chunk:
        sym = sym.strip().upper()
        industry = get_stock_info(sym, st.session_state.nifty_data_df)
        try:
            if page == "Scanner":
                for lbl, tint in active_tfs.items():
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                    m = get_mfi_metrics(hist)
                    if m:
                        if source_mode == "nifty500" and m['passed']:
                            row = {'Symbol': sym, 'Sector': industry, 'Price': m['Price']}
                            if selected_tf_label != "All": row.update({'TF': lbl})
                            st.session_state.processed_results.append(row)
                        elif source_mode == "upload":
                            st.session_state.processed_results.append({
                                'Symbol': sym, 'TF': lbl, 'Sector': industry, 'Price': m['Price'],
                                'Peak_Pk': m['Actual_Peak'], 'Tip': m['Tip'], 'Reached': m['Reached'],
                                'Prox%': m['Prox%'], 'Int%': m['Int%'], 'Date': m['Date']
                            })
            else: # DTE Meter
                for lbl, tint in active_tfs.items():
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                    d = calculate_dte_metrics(hist)
                    if d:
                        st.session_state.processed_results.append({
                            'Symbol': sym, 'TF': lbl, 'Sector': industry, 'Price': d['Price'],
                            'DTE_Price': d['DTE_Price'], 'Reached': d['Reached'], 'Int%': d['Int%'], 'Date': d['Peak_Date']
                        })
        except: pass
        
    st.session_state.last_index += len(chunk)
    if st.session_state.last_index >= len(stock_list):
        st.session_state.is_running = False
    st.rerun()