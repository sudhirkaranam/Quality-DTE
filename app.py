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

st.set_page_config(page_title="Stock DTE & MFI Meter", layout="wide")

# --- Initialize Session State ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'nifty_data_df' not in st.session_state:
    st.session_state.nifty_data_df = pd.DataFrame()

# --- Helper Functions ---

def fetch_nse_master_data():
    """Fetches the official Nifty 500 list."""
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.upper()
        return df
    except Exception:
        return pd.DataFrame()

def get_industry_hybrid(sym, master_df):
    if not master_df.empty and sym in master_df['SYMBOL'].values:
        return master_df[master_df['SYMBOL'] == sym]['INDUSTRY'].values[0]
    try:
        ticker = yf.Ticker(f"{sym}.NS")
        return ticker.info.get('sector', 'N/A')
    except:
        return "N/A"

def get_tip_price(df, target_series_name, target_idx):
    """Calculates the price equivalent for a specific bar's value in a secondary series."""
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
    except:
        return 0

def calculate_combined_metrics(df):
    """Independent DTE (Volume) and MFI metrics with peak dates and percentages."""
    if df is None or df.empty or len(df) < 15: return None
    
    df = df.copy()
    current_price = df['close'].iloc[-1]
    
    # MFI Calculation & Bill Williams Coloring
    df['mfi'] = (df['high'] - df['low']) / df['volume']
    df['mfi_up'] = df['mfi'] > df['mfi'].shift(1)
    df['vol_up'] = df['volume'] > df['volume'].shift(1)
    
    def get_color(row):
        if row['mfi_up'] and row['vol_up']: return "Green"
        if not row['mfi_up'] and not row['vol_up']: return "Fade"
        if row['mfi_up'] and not row['vol_up']: return "Fake"
        return "Squat"
    
    df['mfi_color'] = df.apply(get_color, axis=1)
    max_vol_idx = df['volume'].idxmax()
    max_mfi_idx = df['mfi'].idxmax()

    vol_dte_price = get_tip_price(df, 'volume', max_vol_idx)
    mfi_dte_price = get_tip_price(df, 'mfi', max_mfi_idx)
    mfi_at_vol_peak_price = get_tip_price(df, 'mfi', max_vol_idx)

    return {
        'price': round(current_price, 2),
        'vol_dte': vol_dte_price,
        'vol_gap_pct': round(((vol_dte_price - current_price) / current_price) * 100, 2),
        'vol_peak_date': max_vol_idx.strftime('%Y-%m-%d'),
        'mfi_dte': mfi_dte_price,
        'mfi_gap_pct': round(((mfi_dte_price - current_price) / current_price) * 100, 2),
        'mfi_peak_date': max_mfi_idx.strftime('%Y-%m-%d'),
        'mfi_peak_color': df.loc[max_mfi_idx, 'mfi_color'],
        'mfi_at_vol_peak': mfi_at_vol_peak_price
    }

# Startup
if st.session_state.nifty_data_df.empty:
    st.session_state.nifty_data_df = fetch_nse_master_data()

st.title("📊 Complete DTE & MFI Analytics")

# --- SECTION 1: QUICK LOOKUP ---
st.subheader("🔍 Single Stock Quick Lookup")
quick_sym = st.text_input("Enter NSE Symbol:").strip().upper()

if quick_sym:
    tv_quick = TvDatafeed()
    q_res = []
    mapping = {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly}
    for lbl, tint in mapping.items():
        hist = tv_quick.get_hist(symbol=quick_sym, exchange='NSE', interval=tint, n_bars=100)
        m = calculate_combined_metrics(hist)
        if m:
            q_res.append({
                "Interval": lbl, "Price": m['price'],
                "Vol DTE": m['vol_dte'], "Vol Gap%": m['vol_gap_pct'], "Vol Peak Date": m['vol_peak_date'],
                "MFI Tip": m['mfi_dte'], "MFI Gap%": m['mfi_gap_pct'], "MFI Color": m['mfi_peak_color'], 
                "MFI Date": m['mfi_peak_date'], "MFI @ Vol Peak": m['mfi_at_vol_peak']
            })
    if q_res: st.table(pd.DataFrame(q_res))

st.divider()

# --- SECTION 2: BATCH SCANNER ---
st.subheader("📑 Batch Scanner & Excel Upload")
uploaded_file = st.file_uploader("Upload Excel with 'Symbol' column", type=["xlsx", "xls"])

stock_list = []
if uploaded_file:
    df_in = pd.read_excel(uploaded_file)
    sym_col = next((c for c in df_in.columns if c.lower() == 'symbol'), None)
    if sym_col: stock_list = df_in[sym_col].dropna().astype(str).tolist()
else:
    if st.checkbox("Use NIFTY 500 Index"): 
        stock_list = st.session_state.nifty_data_df['SYMBOL'].tolist() if not st.session_state.nifty_data_df.empty else []

if stock_list:
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🚀 Start/Resume"): st.session_state.is_running = True
    with col2:
        if st.button("Pause"): st.session_state.is_running = False
    with col3:
        if st.button("Reset"):
            st.session_state.processed_results = []; st.session_state.last_index = 0
            st.session_state.is_running = False; st.rerun()

    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        st.dataframe(df_res, use_container_width=True)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='DTE_MFI_Report')
        st.download_button("💾 Download Report", output.getvalue(), "Full_Report.xlsx")

    if st.session_state.is_running and st.session_state.last_index < len(stock_list):
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / len(stock_list))
        while st.session_state.last_index < len(stock_list) and st.session_state.is_running:
            sym = stock_list[st.session_state.last_index].strip().upper()
            row = {'Symbol': sym, 'Industry': get_industry_hybrid(sym, st.session_state.nifty_data_df)}
            for lbl, tint in {'D': Interval.in_daily, 'W': Interval.in_weekly}.items():
                hist = tv.get_hist(symbol=sym, exchange='NSE', interval=tint, n_bars=100)
                m = calculate_combined_metrics(hist)
                if m:
                    row.update({
                        f'{lbl}_Price': m['price'], f'{lbl}_Vol_DTE': m['vol_dte'], f'{lbl}_Vol_Gap%': m['vol_gap_pct'],
                        f'{lbl}_MFI_Tip': m['mfi_dte'], f'{lbl}_MFI_Color': m['mfi_peak_color'], f'{lbl}_MFI_at_VolPeak': m['mfi_at_vol_peak']
                    })
            st.session_state.processed_results.append(row)
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / len(stock_list))
            if st.session_state.last_index % 5 == 0: st.rerun()