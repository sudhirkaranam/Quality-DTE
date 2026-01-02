import streamlit as st
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import yfinance as yf
import io
from datetime import datetime
import requests

st.set_page_config(page_title="Stock Analysis Hub", layout="wide")

# --- 1. INITIALIZE SESSION STATE ---
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

# --- 2. HELPER FUNCTIONS ---
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

def get_mfi_metrics(df):
    if df is None or df.empty or len(df) < 2: return None
    df = df.copy()
    current_price = df['close'].iloc[-1]
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    max_mfi_idx = df['mfi'].idxmax()
    actual_at_peak = df.loc[max_mfi_idx, 'close']
    
    # Calculate Tip Price (Simplified scaling for stability)
    mfi_max = df['mfi'].max()
    mfi_min = df['mfi'].min()
    p_max = df['close'].max()
    p_min = df['close'].min()
    
    if (mfi_max - mfi_min) == 0: return None
    tip_price = p_min + ((mfi_max - mfi_min) / (mfi_max - mfi_min)) * (p_max - p_min)

    prox_pct = abs((current_price - actual_at_peak) / actual_at_peak) * 100
    int_pct = abs((tip_price - actual_at_peak) / actual_at_peak) * 100
    
    # Post-peak reach check
    post_peak_df = df.iloc[df.index.get_loc(max_mfi_idx)+1:]
    reached = "Yes" if not post_peak_df.empty and post_peak_df['close'].max() >= tip_price else "No"
    
    return {
        'Price': round(current_price, 2), 'Actual_Peak': round(actual_at_peak, 2),
        'Tip': round(tip_price, 2), 'Reached': reached,
        'Prox%': round(prox_pct, 2), 'Int%': round(int_pct, 2),
        'Date': max_mfi_idx.strftime('%Y-%m-%d %H:%M'),
        'passed': (prox_pct < 2.0) and (int_pct > 15.0)
    }

# --- 3. NAVIGATION ---
if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

st.sidebar.title("🛠️ Navigation")
page = st.sidebar.radio("Module", ["Scanner", "DTE Meter"])

if page != st.session_state.current_module:
    st.session_state.processed_results = []
    st.session_state.last_index = 0
    st.session_state.is_running = False
    st.session_state.current_module = page
    st.rerun()

# --- 4. BATCH SCANNER UI ---
st.title(f"🚀 {page} Module")

uploaded_file = st.file_uploader("Upload Symbols Excel", type=["xlsx", "xls"])
stock_list = []
source_mode = "manual"

if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
    source_mode = "upload"
elif st.checkbox("Use NIFTY 500 Index"):
    stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist()
    source_mode = "nifty500"

# Control Buttons
c1, c2, c3 = st.columns(3)
if c1.button("▶️ Start Scan"):
    st.session_state.processed_results = []
    st.session_state.last_index = 0
    st.session_state.is_running = True
if c2.button("⏸️ Pause"): st.session_state.is_running = False
if c3.button("🔄 Reset"):
    st.session_state.processed_results = []; st.session_state.last_index = 0
    st.session_state.is_running = False; st.rerun()

# --- 5. RESULTS DISPLAY (Drawn every rerun) ---
if st.session_state.processed_results:
    st.subheader("📊 Scan Results")
    df_res = pd.DataFrame(st.session_state.processed_results)
    st.dataframe(df_res, width=1400)
    
    # Download with Date and TF
    dl_date = datetime.now().strftime("%Y-%m-%d")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_res.to_excel(writer, index=False)
    st.download_button("💾 Download Report", output.getvalue(), f"{page}_Report_{dl_date}.xlsx")

# --- 6. THE PROCESSING ENGINE ---
if st.session_state.is_running and st.session_state.last_index < len(stock_list):
    progress_bar = st.progress(st.session_state.last_index / len(stock_list))
    tv = get_tv_connection()
    
    chunk = stock_list[st.session_state.last_index : st.session_state.last_index + 5]
    
    for sym in chunk:
        sym = sym.strip().upper()
        industry = get_stock_info(sym, st.session_state.nifty_data_df)
        
        try:
            if page == "Scanner":
                for tf_label, tf_val in {'H': Interval.in_1_hour, 'D': Interval.in_daily, 'W': Interval.in_weekly}.items():
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tf_val, n_bars=100)
                    m = get_mfi_metrics(hist)
                    if m:
                        if source_mode == "nifty500" and m['passed']:
                            # ONLY Symbol, Sector, Price for Nifty 500 Matches
                            st.session_state.processed_results.append({
                                'Symbol': sym, 'Sector': industry, 'Price': m['Price'], 'TF': tf_label
                            })
                        elif source_mode == "upload":
                            # FULL Details for Uploaded File
                            st.session_state.processed_results.append({
                                'Symbol': sym, 'TF': tf_label, 'Sector': industry, 'Price': m['Price'],
                                'Peak_Price': m['Actual_Peak'], 'Tip': m['Tip'], 'Reached': m['Reached'],
                                'Prox%': m['Prox%'], 'Int%': m['Int%'], 'Date': m['Date']
                            })
            else: # DTE Meter
                for tf_label, tf_val in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}.items():
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tf_val, n_bars=100)
                    # (DTE logic here... adding Symbol, Sector, Price, TF, DTE_Price)
                    st.session_state.processed_results.append({'Symbol': sym, 'TF': tf_label, 'Sector': industry, 'Price': hist['close'].iloc[-1]})
        except: pass
        
    st.session_state.last_index += len(chunk)
    if st.session_state.last_index >= len(stock_list):
        st.session_state.is_running = False
    st.rerun()