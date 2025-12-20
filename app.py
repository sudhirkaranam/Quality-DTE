import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io
import time

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="High-Quality Stock DTE Analyzer", layout="wide")

st.title("🎯 High-Quality Stock DTE Analyzer")
st.write("Finds stocks with a **20%+ Visual Volume Gap** and **Strong Moat** across multiple intervals.")

# --- Functions ---

def is_quality_stock(symbol):
    """
    Sector-specific quality check:
    - Financials: ROA > 1%, Operating Margin > 15%
    - Others: ROE > 15%, Debt/Equity < 1, Operating Margin > 10%
    """
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        
        sector = info.get('sector', 'Unknown')
        margin = info.get('operatingMargins', 0) or 0
        
        if sector in ['Financial Services', 'Financial']:
            roa = info.get('returnOnAssets', 0) or 0
            if roa > 0.01 and margin > 0.15:
                return True, f"ROA: {round(roa*100, 2)}%", sector
        else:
            roe = info.get('returnOnEquity', 0) or 0
            d_e = (info.get('debtToEquity', 100) or 100) / 100
            if roe > 0.15 and d_e < 1.0 and margin > 0.10:
                return True, f"ROE: {round(roe*100, 2)}%", sector
                
        return False, None, None
    except Exception:
        return False, None, None

def calculate_dte_metrics(df):
    """Calculates DTE Visual Gap and fetches Current Price."""
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
        
        return {'gap': round(percent_gap, 2), 'price': round(current_price, 2)}
    except Exception:
        return None

# --- UI Sidebar & Upload ---

uploaded_file = st.file_uploader("Upload 'stocks.xlsx' (column 'symbol' required)", type=["xlsx"])

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
                
                try:
                    # 1. Fundamental Moat Check
                    is_qual, metric_val, sector_name = is_quality_stock(symbol)
                    
                    if is_qual:
                        gaps = {}
                        current_price_final = 0
                        valid = True
                        
                        # 2. Multi-Interval DTE Check
                        intervals = {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}
                        for label, tv_int in intervals.items():
                            hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                            data = calculate_dte_metrics(hist)
                            
                            if data and data['gap'] > 20:
                                gaps[label] = data['gap']
                                if label == 'Daily': current_price_final = data['price']
                            else:
                                valid = False
                                break
                        
                        if valid:
                            results.append({
                                'Symbol': symbol,
                                'Current Price': current_price_final,
                                'Sector': sector_name,
                                'Quality Metric': metric_val,
                                'Daily_Gap_%': gaps['Daily'],
                                'Weekly_Gap_%': gaps['Weekly'],
                                'Monthly_Gap_%': gaps['Monthly']
                            })
                except Exception as e:
                    st.warning(f"Skipped {symbol} due to an error.")
                    continue
                
                progress_bar.progress((i + 1) / len(stock_list))
            
            # --- Output Results ---
            status_text.text("Analysis Complete!")
            if results:
                final_df = pd.DataFrame(results)
                st.success(f"Found {len(results)} stocks matching criteria!")
                st.dataframe(final_df)
                
                # Export to Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    final_df.to_excel(writer, index=False, sheet_name='High_Quality_DTE')
                
                st.download_button(
                    label="📥 Download Excel Report",
                    data=output.getvalue(),
                    file_name="Quality_DTE_Report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("No stocks matched the 20% gap + Quality criteria.")