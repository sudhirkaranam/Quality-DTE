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

st.set_page_config(page_title="Stock DTE Meter aaa", layout="wide")

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
    """Aggressive search for NSE symbols using Yahoo Finance."""
    results = []
    try:
        # Method 1: Check if the entry itself is a valid NSE ticker
        test_sym = entry.upper().replace(".NS", "")
        ticker_obj = yf.Ticker(f"{test_sym}.NS")
        if ticker_obj.history(period="1d").empty is False:
            results.append({"label": f"Direct Match: {test_sym}", "symbol": test_sym})

        # Method 2: Keyword search
        search = yf.Search(entry, max_results=10)
        if search.tickers:
            for t in search.tickers:
                s = t.get('symbol', '')
                if s.endswith('.NS'):
                    name = t.get('shortname') or t.get('longname') or s
                    results.append({"label": f"{name} ({s})", "symbol": s.replace('.NS', '')})
        
        # Remove duplicates
        seen = set()
        unique_results = []
        for r in results:
            if r['symbol'] not in seen:
                unique_results.append(r)
                seen.add(r['symbol'])
        return unique_results
    except:
        return results

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
user_input = st.text_input("Enter Company Name or Ticker (e.g. 'Reliance' or 'TCS'):").strip()

if user_input:
    matches = search_nse_symbols(user_input)
    if matches:
        options = {m['label']: m['symbol'] for m in matches}
        selected_label = st.selectbox("Select exact stock:", options.keys())
        quick_sym = options[selected_label]
        
        if st.button("Run Analysis"):
            with st.spinner(f"Analyzing {quick_sym}..."):
                tv_quick = TvDatafeed()
                ticker = yf.Ticker(f"{quick_sym}.NS")
                sector = ticker.info.get('sector', 'N/A') if ticker.info else 'N/A'
                q_res = []
                mapping = {'Daily (1D)': Interval.in_daily, 'Weekly (1W)': Interval.in_weekly}
                
                for lbl, tint in mapping.items():
                    hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        q_res.append({"Interval": lbl, "Price": data['price'], "DTE Price": data['dte_lvl'], "Gap%": data['gap']})
            
            if q_res:
                st.info(f"**Symbol:** {quick_sym} | **Sector:** {sector}")
                st.table(pd.DataFrame(q_res))
            else:
                st.warning("No historical data found. TradingView may be offline.")
    else:
        st.error("No NSE matches found. Try using only the ticker symbol (e.g., 'SBIN').")

st.divider()

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📑 Batch Scanner")
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx", "xls"])
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
        if st.button("🚀 Start/Resume"): st.session_state.is_running = True
    with c2:
        if st.button("Pause"): st.session_state.is_running = False
    with c3:
        if st.button("Reset Scanner"):
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
            
            elapsed = time.time() - start_time
            processed_now = (idx - st.session_state.last_index) + 1
            avg_time = elapsed / processed_now if processed_now > 0 else 1.5
            remaining = avg_time * (len(stock_list) - idx)
            status_text.text(f"Scanning {sym}... Est. remaining: {int(remaining // 60)}m {int(remaining % 60)}s")
            
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
        st.download_button("💾 Download Report", output.getvalue(), f"DTE_Report_{datetime.now().strftime('%Y%m%d')}.xlsx")