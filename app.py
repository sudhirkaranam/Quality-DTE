import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="Quality Stock DTE Analyzer", layout="wide")

st.title("🎯 High-Quality Stock DTE Analyzer")
st.write("Upload your stock list, and I'll find stocks with a **20%+ Visual Gap** and **Strong Moat**.")

# --- Functions ---

def is_quality_stock(symbol):
    """Checks for ROE > 15%, Debt/Equity < 1, and Margins > 10%."""
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        roe = info.get('returnOnEquity', 0) or 0
        debt = (info.get('debtToEquity', 100) or 100) / 100
        margin = info.get('operatingMargins', 0) or 0
        
        if roe > 0.15 and debt < 1.0 and margin > 0.10:
            return True, roe, margin
        return False, roe, margin
    except:
        return False, 0, 0

def calculate_dte_metrics(df):
    """Logic to find the visual price peak relative to volume peak."""
    try:
        if df is None or df.empty or len(df) < 5: return None
        current_price = df['close'].iloc[-1]
        
        # Virtual Plotting to get scales
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
        
        return round(percent_gap, 2)
    except:
        return None

# --- UI Sidebar & Upload ---

uploaded_file = st.file_uploader("Upload 'stocks.xlsx' (must have a 'symbol' column)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    if 'symbol' not in df_input.columns:
        st.error("Excel file must contain a column named 'symbol'")
    else:
        stock_list = df_input['symbol'].tolist()
        
        if st.button("🚀 Start Background Processing"):
            tv = TvDatafeed()
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, symbol in enumerate(stock_list):
                status_text.text(f"Analyzing {symbol} ({i+1}/{len(stock_list)})")
                
                # 1. Fundamental Moat Check
                is_qual, roe, margin = is_quality_stock(symbol)
                if is_qual:
                    gaps = {}
                    valid = True
                    
                    # 2. Multi-Interval DTE Check
                    for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                        hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                        gap = calculate_dte_metrics(hist)
                        
                        if gap is not None and gap > 20:
                            gaps[label] = gap
                        else:
                            valid = False
                            break
                    
                    if valid:
                        results.append({
                            'Symbol': symbol,
                            'ROE': f"{round(roe*100, 2)}%",
                            'Margin': f"{round(margin*100, 2)}%",
                            'Daily_Gap_%': gaps['Daily'],
                            'Weekly_Gap_%': gaps['Weekly'],
                            'Monthly_Gap_%': gaps['Monthly']
                        })
                
                progress_bar.progress((i + 1) / len(stock_list))
            
            # --- Output Results ---
            if results:
                final_df = pd.DataFrame(results)
                st.success(f"Found {len(results)} Quality Stocks!")
                st.dataframe(final_df)
                
                # Export to Excel in Memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='Filtered_Stocks')
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name="Quality_DTE_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No stocks matched the 20% gap + Quality criteria.")