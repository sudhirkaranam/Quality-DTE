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

st.set_page_config(page_title="Stock DTE Meter SK", layout="wide")

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
    """Fetches a list of possible NSE symbols based on user entry."""
    try:
        search = yf.Search(entry, max_results=8)
        # Filter for NSE (National Stock Exchange of India) specifically
        results = [
            {"label": f"{t['quotesummary']['longName']} ({t['symbol']})", "symbol": t['symbol'].replace('.NS', '')}
            for t in search.tickers if t['symbol'].endswith('.NS')
        ]
        return results
    except:
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
user_input = st.text_input("Search Company Name (e.g., 'Tata', 'HDFC', 'Reliance'):").strip()

if user_input:
    matches = search_nse_symbols(user_input)
    
    if matches:
        # Create a dropdown for the user to select the correct stock
        options = {m['label']: m['symbol'] for m in matches}
        selected_label = st.selectbox("Select the correct company:", options.keys())
        quick_sym = options[selected_label]
        
        if st.button("Analyze Selected Stock"):
            with st.status(f"Fetching data for {quick_sym}...", expanded=True) as status:
                tv_quick = TvDatafeed()
                ticker = yf.Ticker(f"{quick_sym}.NS")
                sector = ticker.info.get('sector', 'N/A')
                q_res = []
                
                mapping = {'Daily (1D)': Interval.in_daily, 'Weekly (1W)': Interval.in_weekly}
                
                for lbl, tint in mapping.items():
                    hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        q_res.append({"Interval": lbl, "Price": data['price'], "DTE Price": data['dte_lvl'], "Gap%": data['gap']})
                status.update(label="Analysis Complete!", state="complete")

            if q_res:
                st.write(f"**Sector:** {sector} | **Ticker:** {quick_sym}")
                st.table(pd.DataFrame(q_res))
    else:
        st.warning("No matching NSE symbols found. Try refining your search terms.")

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
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("🚀 Start/Resume"): st.session_state.is_running = True
    with c2:
        if st.button("Pause"): st.session_state.is_running = False
    with c3:
        if st.button("Reset"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

    if st.session_state.is_running and st.session_state.last_index < len(stock_list):
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / len(stock_list))
        status_text = st.empty()
        
        proc_map = {'D': Interval.in_daily, 'W': Interval.in_weekly}
        start_time = time.time()
        
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            idx = st.session_state.last_index
            sym = stock_list[idx].strip().upper()
            
            # Time calculation
            elapsed = time.time() - start_time
            processed = idx + 1
            avg_time = elapsed / processed if processed > 0 else 0
            remaining = avg_time * (len(stock_list) - processed)
            status_text.text(f"Scanning {sym}... Est. time remaining: {int(remaining // 60)}m {int(remaining % 60)}s")
            
            try:
                metrics = {}
                ticker = yf.Ticker(f"{sym}.NS")
                sector = ticker.info.get('sector', 'N/A')
                cp = 0
                
                for lbl, tint in proc_map.items():
                    hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                    d = calculate_dte_metrics(hist)
                    if d:
                        metrics[f'{lbl}_Price'] = d['dte_lvl']
                        metrics[f'{lbl}_Gap%'] = d['gap']
                        cp = d['price']

                st.session_state.processed_results.append({
                    'Symbol': sym, 'Sector': sector, 'Price': cp,
                    'D_DTE': metrics.get('D_Price', 0), 'D_Gap%': metrics.get('D_Gap%', 0),
                    'W_DTE': metrics.get('W_Price', 0), 'W_Gap%': metrics.get('W_Gap%', 0)
                })
            except: pass
            
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 5 == 0: st.rerun()

    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        
        st.subheader("🔥 Top 10 Highest Weekly Gaps")
        top_10 = df_res.nlargest(10, 'W_Gap%')[['Symbol', 'Sector', 'Price', 'W_DTE', 'W_Gap%']]
        st.dataframe(top_10, use_container_width=True)
        
        st.subheader("📋 All Results")
        st.dataframe(df_res, use_container_width=True)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='DTE_Report')
        st.download_button("💾 Download Full Report", output.getvalue(), f"DTE_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")