import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io

# Setup non-interactive plotting
matplotlib.use('Agg')

st.set_page_config(page_title="Ultimate Stock DTE & Quality Analyzer", layout="wide")

st.title("📊 Comprehensive Stock Analyzer")
st.write("Processing all stocks with full Technical DTE and Fundamental Quality data.")

# --- Helper Functions ---

def get_mcap_category(mcap):
    """Categorizes stock based on Market Cap (INR Cr)."""
    if not mcap: return "Unknown"
    mcap_cr = mcap / 10000000  # Convert to Crores
    if mcap_cr >= 20000: return "Large Cap"
    elif mcap_cr >= 5000: return "Mid Cap"
    else: return "Small Cap"

def calculate_dte_metrics(df):
    """Returns gap and the actual DTE Price level."""
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

uploaded_file = st.file_uploader("Upload 'stocks.xlsx' (column 'symbol' required)", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    stock_list = df_input['symbol'].tolist()
    
    if st.button("🚀 Run Full Analysis"):
        tv = TvDatafeed()
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(stock_list):
            status_text.text(f"Processing {symbol} ({i+1}/{len(stock_list)})")
            try:
                # 1. Fetch Fundamentals (No filtering, just data collection)
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                
                sector = info.get('sector', 'Unknown')
                mcap = info.get('marketCap')
                mcap_cat = get_mcap_category(mcap)
                
                # Moat Metrics
                roe = info.get('returnOnEquity', 0)
                roa = info.get('returnOnAssets', 0)
                debt_eq = info.get('debtToEquity', 0)
                op_margin = info.get('operatingMargins', 0)
                
                # 2. Fetch Technicals
                gaps, dte_lvls, curr_p = {}, {}, 0
                for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                    hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        gaps[label] = data['gap']
                        dte_lvls[label] = data['dte_lvl']
                        if label == 'Daily': curr_p = data['price']

                # 3. Combine Data into Row
                results.append({
                    'Symbol': symbol,
                    'Current Price': curr_p,
                    'Cap Category': mcap_cat,
                    'Sector': sector,
                    'ROE%': round(roe * 100, 2) if roe else 0,
                    'ROA%': round(roa * 100, 2) if roa else 0,
                    'Debt/Equity': round(debt_eq / 100, 2) if debt_eq else 0,
                    'Op Margin%': round(op_margin * 100, 2) if op_margin else 0,
                    'D_DTE_Price': dte_lvls.get('Daily', 0),
                    'D_Gap%': gaps.get('Daily', 0),
                    'W_DTE_Price': dte_lvls.get('Weekly', 0),
                    'W_Gap%': gaps.get('Weekly', 0),
                    'M_DTE_Price': dte_lvls.get('Monthly', 0),
                    'M_Gap%': gaps.get('Monthly', 0)
                })

            except Exception as e:
                # Skip stocks with errors but keep the app running
                continue
            
            progress_bar.progress((i + 1) / len(stock_list))

        status_text.text("Analysis Complete!")
        
        if results:
            final_df = pd.DataFrame(results)
            st.dataframe(final_df)
            
            # Export to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False, sheet_name='Stock_Analysis')
            
            st.download_button(
                label="📥 Download Full Report",
                data=output.getvalue(),
                file_name="Full_Stock_Analysis_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.error("Could not fetch data for the symbols provided.")