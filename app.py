import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io
from datetime import datetime
import time

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

# --- UI Controls ---

uploaded_file = st.file_uploader("Upload 'stocks.xlsx'", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    stock_list = df_input['symbol'].tolist()
    total_stocks = len(stock_list)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start / Resume Scan"):
            st.session_state.is_running = True
    
    with col2:
        if st.button("⏸️ Pause Scan"):
            st.session_state.is_running = False
            st.rerun()

    with col3:
        if st.button("🗑️ Reset Everything"):
            st.session_state.processed_results = []
            st.session_state.last_index = 0
            st.session_state.is_running = False
            st.rerun()

    # --- Processing Engine ---
    if st.session_state.is_running and st.session_state.last_index < total_stocks:
        tv = TvDatafeed()
        progress_bar = st.progress(st.session_state.last_index / total_stocks)
        status_text = st.empty()
        
        # We process in a loop without st.rerun() inside to keep it moving
        while st.session_state.last_index < total_stocks and st.session_state.is_running:
            i = st.session_state.last_index
            symbol = stock_list[i]
            status_text.text(f"Currently Analyzing: {symbol} ({i+1}/{total_stocks})")
            
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                
                # Fundamental Data
                fcf = info.get('freeCashflow', 0) or 0
                total_debt = info.get('totalDebt', 0) or 0
                total_equity = info.get('totalStockholderEquity', 0) or info.get('bookValue', 0) * info.get('sharesOutstanding', 1)
                invested_capital = total_debt + total_equity
                croic = (fcf / invested_capital) if invested_capital > 0 else 0

                # Technical Data
                gaps, dte_lvls, curr_p = {}, {}, 0
                for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                    hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        gaps[label] = data['gap']
                        dte_lvls[label] = data['dte_lvl']
                        if label == 'Daily': curr_p = data['price']

                # Append to persistent state
                st.session_state.processed_results.append({
                    'Symbol': symbol,
                    'Current Price': curr_p,
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Book Price': round(info.get('bookValue', 0) or 0, 2),
                    'CROIC %': round(croic * 100, 2),
                    'ROE%': round((info.get('returnOnEquity', 0) or 0) * 100, 2),
                    'Debt/Equity': round((info.get('debtToEquity', 0) or 0) / 100, 2),
                    'Op Margin%': round((info.get('operatingMargins', 0) or 0) * 100, 2),
                    'D_DTE_Price': dte_lvls.get('Daily', 0), 'D_Gap%': gaps.get('Daily', 0),
                    'W_DTE_Price': dte_lvls.get('Weekly', 0), 'W_Gap%': gaps.get('Weekly', 0),
                    'M_DTE_Price': dte_lvls.get('Monthly', 0), 'M_Gap%': gaps.get('Monthly', 0)
                })
            except Exception:
                pass
            
            # Increment and update UI
            st.session_state.last_index += 1
            progress_bar.progress(st.session_state.last_index / total_stocks)
            
            # Periodic UI refresh to keep browser alive without stopping the loop
            if st.session_state.last_index % 10 == 0:
                st.rerun()

        if st.session_state.last_index >= total_stocks:
            st.session_state.is_running = False
            st.success("Analysis Complete!")

    # --- Result Display & Excel Download ---
    if st.session_state.processed_results:
        df_res = pd.DataFrame(st.session_state.processed_results)
        st.write(f"### Results ({len(df_res)} stocks)")
        st.dataframe(df_res)
        
        # Formatted Excel Export
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_res.to_excel(writer, index=False, sheet_name='Report')
            workbook = writer.book
            worksheet = writer.sheets['Report']
            worksheet.freeze_panes(1, 1)
            
            # Formatting
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            for col_num, value in enumerate(df_res.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
            
            worksheet.set_column('A:A', 15)
            worksheet.set_column('B:G', 15)

        st.download_button("📥 Download Excel Report", output.getvalue(), f"Stock_DTE_{datetime.now().strftime('%Y%m%d')}.xlsx")