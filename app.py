import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="Comprehensive Stock Analyzer", layout="wide")

st.title("📊 Ultimate Stock Analyzer")
st.write("Technical DTE Levels + Deep Fundamental Quality Metrics")

# --- Helper Functions ---

def get_mcap_category(mcap):
    if not mcap: return "Unknown"
    mcap_cr = mcap / 10000000 
    if mcap_cr >= 20000: return "Large Cap"
    elif mcap_cr >= 5000: return "Mid Cap"
    else: return "Small Cap"

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
    except:
        return None

# --- UI Sidebar & Upload ---

uploaded_file = st.file_uploader("Upload 'stocks.xlsx'", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    stock_list = df_input['symbol'].tolist()
    
    if st.button("🚀 Run Analysis"):
        tv = TvDatafeed()
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stock_list):
            status_text.text(f"Processing {symbol} ({i+1}/{len(stock_list)})")
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                
                # Fundamental Data
                sector = info.get('sector', 'N/A')
                industry = info.get('industry', 'N/A')
                mcap_cat = get_mcap_category(info.get('marketCap'))
                book_val = info.get('bookValue', 0)
                
                # CROIC Calculation
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

                results.append({
                    'Symbol': symbol,
                    'Current Price': curr_p,
                    'Cap Category': mcap_cat,
                    'Sector': sector,
                    'Industry': industry,
                    'Book Price': round(book_val, 2),
                    'CROIC %': round(croic * 100, 2),
                    'ROE%': round(info.get('returnOnEquity', 0) * 100, 2),
                    'Debt/Equity': round((info.get('debtToEquity', 0) or 0) / 100, 2),
                    'D_DTE': dte_lvls.get('Daily', 0), 'D_Gap%': gaps.get('Daily', 0),
                    'W_DTE': dte_lvls.get('Weekly', 0), 'W_Gap%': gaps.get('Weekly', 0),
                    'M_DTE': dte_lvls.get('Monthly', 0), 'M_Gap%': gaps.get('Monthly', 0)
                })

            except Exception:
                continue
            progress_bar.progress((i + 1) / len(stock_list))

        if results:
            final_df = pd.DataFrame(results)
            st.dataframe(final_df)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 Download Full Report", output.getvalue(), "Stock_Analysis.xlsx")