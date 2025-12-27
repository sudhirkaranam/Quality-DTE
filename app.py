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

st.set_page_config(page_title="Stock DTE Meter 11", layout="wide")

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
    except Exception:
        return []

def search_nse_symbols(entry):
    """Robustly fetches NSE symbols based on user entry."""
    try:
        # Use yf.Search but with safer key access
        search = yf.Search(entry, max_results=10)
        results = []
        
        if search.tickers:
            for t in search.tickers:
                symbol = t.get('symbol', '')
                # Ensure we only get Indian stocks from NSE
                if symbol.endswith('.NS'):
                    # Try to get a name, fallback to symbol if name is missing
                    name = t.get('shortname') or t.get('longname') or symbol
                    results.append({
                        "label": f"{name} ({symbol})", 
                        "symbol": symbol.replace('.NS', '')
                    })
        return results
    except Exception as e:
        return []

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

if not st.session_state.nifty_500_list:
    st.session_state.nifty_500_list = get_nifty_500()

st.title("📊 Stock DTE Meter")

# --- SECTION 1: SEARCH & QUICK LOOKUP ---
st.subheader("🔍 Single Stock Search")
user_input = st.text_input("Search Company Name (e.g., 'Tata Motors', 'HDFC Bank'):").strip()

if user_input:
    matches = search_nse_symbols(user_input)
    
    if matches:
        options = {m['label']: m['symbol'] for m in matches}
        selected_label = st.selectbox("Select the correct company from results:", options.keys())
        quick_sym = options[selected_label]
        
        if st.button("Run DTE Analysis"):
            with st.spinner(f"Analyzing {quick_sym}..."):
                tv_quick = TvDatafeed()
                ticker = yf.Ticker(f"{quick_sym}.NS")
                # Handle potential sector retrieval issues
                sector = ticker.info.get('sector', 'N/A') if ticker.info else 'N/A'
                q_res = []
                
                mapping = {'Daily (1D)': Interval.in_daily, 'Weekly (1W)': Interval.in_weekly}
                
                for lbl, tint in mapping.items():
                    hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        q_res.append({
                            "Interval": lbl, 
                            "Price": data['price'], 
                            "DTE Price": data['dte_lvl'], 
                            "Gap%": data['gap']
                        })
            
            if q_res:
                st.info(f"**Symbol:** {quick_sym} | **Sector:** {sector}")
                st.table(pd.DataFrame(q_res))
            else:
                st.warning("Data fetch failed. TradingView might be blocking the request. Try again in a moment.")
    else:
        st.error("No NSE matches found. Try entering the exact symbol (e.g., 'RELIANCE' instead of 'Reliance Industries').")

st.divider()

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📑 Batch Scanner")
uploaded_file = st.file_uploader("Upload Excel with 'Symbol' column", type=["xlsx", "xls"])
stock_list = []
if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
else:
    if st.checkbox("Use NIFTY 500 Index"): stock_list = st.session_state.nifty_500_list

if stock_list:
    c1, c2, c3 = st.columns(3)
    with c1: