import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import yfinance as yf
from tvDatafeed import TvDatafeed, Interval
import io

matplotlib.use('Agg')
st.set_page_config(page_title="Stock Analyzer", layout="wide")

st.title("🎯 High-Quality Stock DTE Analyzer")

def is_quality_stock(symbol):
    try:
        ticker = yf.Ticker(f"{symbol}.NS")
        info = ticker.info
        sector = info.get('sector', 'Unknown')
        margin = info.get('operatingMargins', 0) or 0
        
        # Checking if 'Financial' is in the sector string (more robust)
        if "Financial" in sector:
            roa = info.get('returnOnAssets', 0) or 0
            # Bank/NBFC: ROA > 0.8% and Margin > 10% (Slightly relaxed for results)
            if roa > 0.008 and margin > 0.10:
                return True, f"ROA: {round(roa*100, 2)}%", sector
        else:
            roe = info.get('returnOnEquity', 0) or 0
            d_e = (info.get('debtToEquity', 100) or 100) / 100
            # Others: ROE > 12%, Debt < 1.2
            if roe > 0.12 and d_e < 1.2 and margin > 0.08:
                return True, f"ROE: {round(roe*100, 2)}%", sector
                
        return False, "Failed Fundamentals", sector
    except:
        return False, "API Error", "Unknown"

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
        return {'gap': round(percent_gap, 2), 'price': round(current_price, 2)}
    except:
        return None

uploaded_file = st.file_uploader("Upload 'stocks.xlsx'", type=["xlsx"])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    stock_list = df_input['symbol'].tolist()
    
    if st.button("🚀 Run Analysis"):
        tv = TvDatafeed()
        results = []
        progress_bar = st.progress(0)
        
        for i, symbol in enumerate(stock_list):
            # 1. Fundamental Check
            is_qual, metric_val, sector_name = is_quality_stock(symbol)
            
            if is_qual:
                gaps = {}
                curr_p = 0
                # 2. Technical Check
                for label, tv_int in {'Daily': Interval.in_daily, 'Weekly': Interval.in_weekly, 'Monthly': Interval.in_monthly}.items():
                    hist = tv.get_hist(symbol=symbol, exchange='NSE', interval=tv_int, n_bars=100)
                    data = calculate_dte_metrics(hist)
                    if data:
                        gaps[label] = data['gap']
                        if label == 'Daily': curr_p = data['price']
                
                # REVISED FILTER: Any interval > 20% (instead of all 3)
                if any(v > 20 for v in gaps.values()):
                    results.append({
                        'Symbol': symbol,
                        'Price': curr_p,
                        'Sector': sector_name,
                        'Metric': metric_val,
                        'D_Gap%': gaps.get('Daily', 0),
                        'W_Gap%': gaps.get('Weekly', 0),
                        'M_Gap%': gaps.get('Monthly', 0)
                    })
            
            progress_bar.progress((i + 1) / len(stock_list))

        if results:
            st.write(pd.DataFrame(results))
            # (Excel Download code remains same as previous step)
        else:
            st.error("Still no matches. Try reducing the 20% gap requirement to 10% in the code.")