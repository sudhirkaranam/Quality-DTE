import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tvDatafeed import TvDatafeed, Interval
import io
from datetime import datetime
import requests

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

st.title("🎯 Stock DTE Meter")

# --- Helper Functions ---

def calculate_dte_metrics(df):
    try:
        if df is None or df.empty or len(df) < 5: return None
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

def get_nifty_500():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        symbol_col = next((c for c in df.columns if c.lower() == 'symbol'), None)
        return df[symbol_col].tolist() if symbol_col else []
    except Exception as e:
        st.error(f"Failed to fetch NIFTY 500: {e}")
        return []

# --- SECTION 1: QUICK LOOKUP (SINGLE STOCK) ---
st.subheader("🔍 Single Stock Quick Lookup")
quick_sym = st.text_input("Enter Stock Symbol (e.g. SBIN, RELIANCE, TCS):").strip().upper()

if quick_sym:
    with st.spinner(f"Fetching DTE for {quick_sym}..."):
        tv_quick = TvDatafeed()
        quick_results = []
        for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
            hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tv_int, n_bars=100)
            data = calculate_dte_metrics(hist)
            if data:
                quick_results.append({
                    "Interval": label,
                    "Current Price": data['price'],
                    "DTE Price Level": data['dte_lvl'],
                    "Gap %": data['gap']
                })
        
        if quick_results:
            st.table(pd.DataFrame(quick_results))
        else:
            st.warning("No data found. Check the symbol and try again.")

st.markdown("---")

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📂 Batch Scanner")
uploaded_file = st.file_uploader("Upload an Excel file (Optional)", type=["xlsx", "xls"])

stock_list = []
if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    symbol_col = next((c for c in df_input.columns if c.lower() == 'symbol'), None)
    if symbol_col:
        stock_list = df_input[symbol_col].dropna().astype(str).tolist()
    else:
        st.error("Error: Could not find a column named 'Symbol'.")
else:
    if st.checkbox("Use NIFTY 500 List", value=False):
        stock_list = get_nifty_500()

if stock_list:
    total_stocks = len(stock_list)
    st.info(f"List loaded: {total_stocks} stocks.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Start/Resume Scan"): st.session_state.is_running = True
    with c2:
        if st.button("Pause Scan"): st.session_state.is_running = False
    with c3:
        if st.button("Reset Scanner"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

    if st.session_state.is_running and st.session_state.last_index < total_stocks:
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / total_stocks)
        while st.session_state.last_index < total_stocks and st.session_state.is_running:
            i = st.session_state.last_index
            symbol = stock_list[i].strip().upper()
            try:
                gaps, dte_lvls, curr_p = {}, {}, 0
                for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                    hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        gaps[label], dte_lvls[label] = data['gap'], data['dte_lvl']
                        if label == 'Daily': curr_p = data['price']

                st.session_state.processed_results.append({
                    'Symbol': symbol, 'Current Price': curr_p,
                    'D_DTE': dte_lvls.get('Daily', 0), 'D_Gap %': gaps.get('Daily', 0),
                    'W_DTE': dte_lvls.get('Weekly', 0), 'W_Gap %': gaps.get('Weekly', 0),
                    'M_DTE': dte_lvls.get('Monthly', 0), 'M_Gap %': gaps.get('Monthly', 0)
                })
            except: pass
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / total_stocks)
            if st.session_state.last_index % 10 == 0: st.rerun()
        
        if st.session_state.last_index >= total_stocks:
            st.session_state.is_running = False
            st.success("Batch Complete!")

    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        st.dataframe(df_res)
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='DTE_Meter')
            workbook = writer.book
            worksheet = writer.sheets['DTE_Meter']
            worksheet.freeze_panes(1, 1)
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            for col_num, value in enumerate(df_res.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
            worksheet.set_column('A:H', 15)
        st.download_button("📥 Download Report", output.getvalue(), f"Stock_DTE_Meter_{datetime.now().strftime('%Y%m%d')}.xlsx")