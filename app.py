import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io
from datetime import datetime

# Setup non-interactive plotting to prevent memory leaks
matplotlib.use('Agg')

st.set_page_config(page_title="Ultimate Stock Analyzer", layout="wide")

# --- Initialize Session State for Batch Memory ---
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = []
if 'last_index' not in st.session_state:
    st.session_state.last_index = 0

st.title("🎯 Stock DTE")
st.write("DTE + Fundamental Quality")

# --- Helper Functions ---

def calculate_dte_metrics(df):
    """Calculates DTE Visual Gap and fetches actual Price levels."""
    try:
        if df is None or df.empty or len(df) < 5: return None
        current_price = df['close'].iloc[-1]
        
        # Virtual Plotting to capture scales
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
        
        return {
            'gap': round(percent_gap, 2), 
            'price': round(current_price, 2), 
            'dte_lvl': round(dte_price, 2)
        }
    except:
        return None

# --- UI Controls ---

uploaded_file = st.file_uploader("Upload 'stocks.xlsx' (column 'symbol' required)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    stock_list = df_input['symbol'].tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Start / Resume Analysis"):
            tv = TvDatafeed()
            progress_bar = st.progress(st.session_state.last_index / len(stock_list))
            status_text = st.empty()
            
            for i in range(st.session_state.last_index, len(stock_list)):
                symbol = stock_list[i]
                status_text.text(f"Analyzing {symbol} ({i+1}/{len(stock_list)})")
                
                try:
                    ticker = yf.Ticker(f"{symbol}.NS")
                    info = ticker.info
                    
                    # 1. Fundamental Calculations
                    fcf = info.get('freeCashflow', 0) or 0
                    total_debt = info.get('totalDebt', 0) or 0
                    total_equity = info.get('totalStockholderEquity', 0) or info.get('bookValue', 0) * info.get('sharesOutstanding', 1)
                    invested_capital = total_debt + total_equity
                    croic = (fcf / invested_capital) if invested_capital > 0 else 0

                    # 2. Multi-Interval Technical Data
                    gaps, dte_lvls, curr_p = {}, {}, 0
                    for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                        hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                        data = calculate_dte_metrics(hist)
                        if data:
                            gaps[label] = data['gap']
                            dte_lvls[label] = data['dte_lvl']
                            if label == 'Daily': curr_p = data['price']

                    # 3. Store in Session State (Market Cap Column Removed)
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
                        'D_DTE_Price': dte_lvls.get('Daily', 0),
                        'D_Gap%': gaps.get('Daily', 0),
                        'W_DTE_Price': dte_lvls.get('Weekly', 0),
                        'W_Gap%': gaps.get('Weekly', 0),
                        'M_DTE_Price': dte_lvls.get('Monthly', 0),
                        'M_Gap%': gaps.get('Monthly', 0)
                    })
                except Exception:
                    continue
                
                st.session_state.last_index = i + 1
                progress_bar.progress(st.session_state.last_index / len(stock_list))
                
                if i % 5 == 0:
                    st.rerun()

    with col2:
        if st.button("🗑️ Reset All Data"):
            st.session_state.processed_results = []
            st.session_state.last_index = 0
            st.rerun()

    # --- Data Display & Mobile-Formatted Export ---
    if st.session_state.processed_results:
        final_df = pd.DataFrame(st.session_state.processed_results)
        st.subheader(f"📊 Processed {len(final_df)} stocks")
        st.dataframe(final_df)
        
        report_time = datetime.now().strftime("%d-%m-%Y %H:%M")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Analysis_Report')
            
            workbook  = writer.book
            worksheet = writer.sheets['Analysis_Report']
            
            # Freeze Top Row and Symbol Column
            worksheet.freeze_panes(1, 1)
            
            # Format and Column Widths
            worksheet.set_column('A:A', 15) # Symbol
            worksheet.set_column('B:G', 15) # Price to Op Margin
            header_fmt = workbook.add_format({'bold': True, 'bg_color': '#D7E4BC', 'border': 1})
            time_fmt = workbook.add_format({'italic': True, 'font_color': 'red'})

            for col_num, value in enumerate(final_df.columns.values):
                worksheet.write(0, col_num, value, header_fmt)
            
            worksheet.write(len(final_df) + 2, 0, f"Last Updated: {report_time}", time_fmt)

        st.download_button(
            label="📥 Download Formatted Excel",
            data=output.getvalue(),
            file_name=f"DTE_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )