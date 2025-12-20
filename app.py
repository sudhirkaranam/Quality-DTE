import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io

matplotlib.use('Agg')
st.set_page_config(page_title="Stock DTE Analyzer", layout="wide")

st.title("🎯 Stock DTE Analyzer")

# --- Helper Functions ---

def get_mcap_category(mcap):
    """Categorizes stock based on Market Cap (INR Cr)."""
    if not mcap: return "Unknown"
    mcap_cr = mcap / 10000000  # Convert to Crores
    if mcap_cr >= 20000: return "Large Cap"
    elif mcap_cr >= 5000: return "Mid Cap"
    else: return "Small Cap"

def is_quality_stock(info):
    """Sector-specific quality check logic."""
    try:
        sector = info.get('sector', 'Unknown')
        margin = info.get('operatingMargins', 0) or 0
        if "Financial" in sector:
            roa = info.get('returnOnAssets', 0) or 0
            if roa > 0.008 and margin > 0.10:
                return True, f"ROA: {round(roa*100, 2)}%", sector
        else:
            roe = info.get('returnOnEquity', 0) or 0
            d_e = (info.get('debtToEquity', 100) or 100) / 100
            if roe > 0.12 and d_e < 1.2 and margin > 0.08:
                return True, f"ROE: {round(roe*100, 2)}%", sector
        return False, None, sector
    except:
        return False, None, "Unknown"

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

# --- UI Controls ---

analysis_type = st.radio("Select Analysis Type:", ["Quality DTE (Filtered)", "All Stocks DTE (Unfiltered)"])
uploaded_file = st.file_uploader("Upload 'stocks.xlsx'", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    stock_list = df_input['symbol'].tolist()
    
    if st.button("🚀 Run Analysis"):
        tv = TvDatafeed()
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(stock_list):
            try:
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                mcap_cat = get_mcap_category(info.get('marketCap'))
                
                # Logic for Quality Filter
                is_qual = True
                metric_val = "N/A"
                sector_name = info.get('sector', 'Unknown')
                
                if analysis_type == "Quality DTE (Filtered)":
                    is_qual, metric_val, sector_name = is_quality_stock(info)
                
                if is_qual:
                    gaps, dte_lvls, curr_p = {}, {}, 0
                    for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                        hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                        data = calculate_dte_metrics(hist)
                        if data:
                            gaps[label] = data['gap']
                            dte_lvls[label] = data['dte_lvl']
                            if label == 'Daily': curr_p = data['price']
                    
                    # Store Result
                    res_row = {
                        'Symbol': symbol,
                        'Current Price': curr_p,
                        'M-Cap Category': mcap_cat,
                        'Sector': sector_name,
                    }
                    
                    if analysis_type == "Quality DTE (Filtered)":
                        res_row['Metric'] = metric_val
                        res_row['D_Gap%'] = gaps.get('Daily', 0)
                        res_row['W_Gap%'] = gaps.get('Weekly', 0)
                        res_row['M_Gap%'] = gaps.get('Monthly', 0)
                        # Filter: At least one interval > 20%
                        if any(v > 20 for v in gaps.values()):
                            results.append(res_row)
                    else:
                        # Unfiltered: Add DTE Prices
                        res_row.update({
                            'D_Price': dte_lvls.get('Daily', 0),
                            'D_Gap%': gaps.get('Daily', 0),
                            'W_Price': dte_lvls.get('Weekly', 0),
                            'W_Gap%': gaps.get('Weekly', 0),
                            'M_Price': dte_lvls.get('Monthly', 0),
                            'M_Gap%': gaps.get('Monthly', 0)
                        })
                        results.append(res_row)

            except Exception:
                continue
            progress_bar.progress((i + 1) / len(stock_list))

        if results:
            final_df = pd.DataFrame(results)
            st.dataframe(final_df)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                final_df.to_excel(writer, index=False)
            st.download_button("📥 Download Report", output.getvalue(), "Stock_DTE_Report.xlsx")
        else:
            st.error("No results found. Check symbols or loosen filters.")