import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import urllib.request
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz

from transformers import pipeline
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score # NEW: VALIDATION METRICS
from hmmlearn.hmm import GaussianHMM
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
st.set_page_config(page_title="Institutional Market Scanner", layout="wide", page_icon="📈")
st.title("⚡ Institutional 24/7 Market Scanner")
st.markdown("**Architecture:** 10-Year Historical AI + Live 1-Minute Intraday Analysis + Time-Series Validation")

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

finbert = load_finbert()

# ==========================================
# 2. DUAL-TIMEFRAME DATA ENGINES
# ==========================================
def fetch_market_data(ticker):
    try:
        stock = yf.Ticker(ticker).history(period="10y")
        vix = yf.Ticker("^VIX").history(period="10y")
        
        if stock.empty:
            st.error(f"❌ ERROR: Yahoo Finance returned no data for {ticker}.")
            st.stop()
            
        stock.index = pd.to_datetime(stock.index).tz_localize(None).normalize()
        vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
        
        df = pd.DataFrame(index=stock.index)
        df['Close'] = stock['Close'].values.astype(float).flatten()
        df['Open'] = stock['Open'].values.astype(float).flatten()
        df['High'] = stock['High'].values.astype(float).flatten()
        df['Low'] = stock['Low'].values.astype(float).flatten()
        
        df['VIX'] = vix['Close']
        df['VIX'] = df['VIX'].ffill().bfill().fillna(20.0) 
        
        return df
    except Exception as e:
        st.error(f"❌ Data Fetch Error: {str(e)}")
        st.stop()

def fetch_and_analyze_intraday(ticker):
    try:
        intraday = yf.Ticker(ticker).history(period="1d", interval="1m")
        if intraday.empty:
            return None
            
        ny_time = datetime.datetime.now(pytz.timezone('US/Eastern'))
        last_data_date = intraday.index[-1].date()
        
        if last_data_date != ny_time.date():
            return None

        intraday['RSI_1m'] = RSIIndicator(close=intraday['Close'], window=14).rsi()
        macd_1m = MACD(close=intraday['Close'])
        intraday['MACD_1m'] = macd_1m.macd()
        intraday['MACD_Signal_1m'] = macd_1m.macd_signal()
        
        return intraday
    except Exception:
        return None

def compute_technical_features(df):
    data = df.copy()
    
    series_vals = data['Close'].values
    res = np.full_like(series_vals, np.nan, dtype=float)
    window = 60
    weights = [1.0]
    for k in range(1, window):
        weights.append(-weights[-1] * (0.5 - k + 1) / k)
    weights = np.array(weights)[::-1]
    
    for i in range(window - 1, len(series_vals)):
        res[i] = np.dot(weights, series_vals[i - window + 1 : i + 1])
        
    data['Frac_Diff_Close'] = res
    data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
    macd = MACD(close=data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_Signal'] = macd.macd_signal()
    
    data['MACD_Bullish_Cross'] = (data['MACD'] > data['MACD_Signal']) & (data['MACD'].shift(1) <= data['MACD_Signal'].shift(1))
    data['MACD_Bearish_Cross'] = (data['MACD'] < data['MACD_Signal']) & (data['MACD'].shift(1) >= data['MACD_Signal'].shift(1))
    
    bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
    data['BB_High'] = bb.bollinger_hband()
    data['BB_Low'] = bb.bollinger_lband()
    
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    data.dropna(inplace=True)
    
    if len(data) < 50:
        st.error(f"❌ ERROR: Not enough data to train.")
        st.stop()
        
    return data

# ==========================================
# 3. LIVE SENTIMENT SCRAPING
# ==========================================
def fetch_live_sentiment(ticker):
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0'}
        req = urllib.request.Request(url, headers=headers)
        html = urllib.request.urlopen(req, timeout=5).read()
        soup = BeautifulSoup(html, 'html.parser')
        
        news_table = soup.find(id='news-table')
        if not news_table:
            return ["No news found."], [{"label": "neutral", "score": 0.0}], 0
            
        headlines = [row.a.text for row in news_table.findAll('tr')][:5] 
        sentiments = finbert(headlines)
        
        score = sum([s['score'] if s['label'] == 'positive' else -s['score'] for s in sentiments if s['label'] != 'neutral'])
        return headlines, sentiments, score
    except Exception:
        return ["⚠️ Scraper Blocked"], [{"label": "neutral", "score": 0.0}], 0

# ==========================================
# 4. AI ENSEMBLE & ACADEMIC VALIDATION
# ==========================================
def train_ai_ensemble(df):
    features = ['Frac_Diff_Close', 'RSI', 'MACD', 'VIX']
    X = df[features].values
    y = df['Target'].values
    
    # --- STRICT TIME-SERIES SPLIT (70% Train, 30% Test) ---
    split_idx = int(len(df) * 0.7)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) # Crucial: Transform test data using Train scaler
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    # --- ACADEMIC VALIDATION SCORING ---
    y_pred = xgb_model.predict(X_test_scaled)
    val_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0)
    }
    
    # Train Hidden Markov Model (Risk Engine)
    returns = np.diff(np.log(df['Close'].values[:split_idx]), prepend=0).reshape(-1, 1)
    hmm_model = GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
    hmm_model.fit(returns)
    variances = [np.diag(hmm_model.covars_[i]) for i in range(2)]
    crash_state = np.argmax(variances)
    
    # Train Meta-Learner
    meta_learner = LogisticRegression()
    xgb_train_preds = xgb_model.predict_proba(X_train_scaled)[:, 1]
    simulated_sentiment = np.random.normal(0, 0.5, len(xgb_train_preds))
    meta_X = np.column_stack((xgb_train_preds, simulated_sentiment))
    meta_learner.fit(meta_X, y_train)
    
    return scaler, xgb_model, hmm_model, crash_state, meta_learner, val_metrics

# ==========================================
# 5. STREAMLIT FRONTEND & EXECUTION
# ==========================================
col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("⚙️ System Controls")
    ticker = st.text_input("Enter Ticker (e.g., AAPL, NVDA)", "NVDA").upper()
    run_scanner = st.button("Run Real-Time Scan", type="primary")

if run_scanner:
    with st.spinner("Fetching 10-Year History & 1-Minute Live Data..."):
        raw_data = fetch_market_data(ticker)
        intraday_data = fetch_and_analyze_intraday(ticker)
        processed_data = compute_technical_features(raw_data)
        
    with st.spinner("Training & Validating AI on 10-Year Dataset..."):
        scaler, xgb_model, hmm_model, crash_state, meta_learner, val_metrics = train_ai_ensemble(processed_data)
        
    with st.spinner("Reading Live Breaking News..."):
        news, sentiment_details, sentiment_score = fetch_live_sentiment(ticker)
        
    latest_data = processed_data.iloc[-1]
    X_live = scaler.transform([latest_data[['Frac_Diff_Close', 'RSI', 'MACD', 'VIX']].values])
    xgb_prob = xgb_model.predict_proba(X_live)[0][1]
    
    meta_X_live = np.column_stack((xgb_prob, sentiment_score))
    final_prob = meta_learner.predict_proba(meta_X_live)[0][1]
    
    recent_returns = np.diff(np.log(raw_data['Close'].values[-10:]), prepend=0).reshape(-1, 1)
    current_regime = hmm_model.predict(recent_returns)[-1]
    
    if current_regime == crash_state:
        final_signal, signal_color = "🛑 HOLD (HMM Detected Crash Regime)", "red"
    elif final_prob > 0.55:
        final_signal, signal_color = "✅ BUY SIGNAL (Ensemble Confirmed)", "green"
    elif final_prob < 0.45:
        final_signal, signal_color = "🔻 SELL / SHORT (Ensemble Confirmed)", "orange"
    else:
        final_signal, signal_color = "⏸️ NEUTRAL / HOLD", "gray"

    hist_data = processed_data.tail(90)
    X_hist_scaled = scaler.transform(hist_data[['Frac_Diff_Close', 'RSI', 'MACD', 'VIX']].values)
    hist_probs = xgb_model.predict_proba(X_hist_scaled)[:, 1]
    
    buy_dates, buy_prices, sell_dates, sell_prices = [], [], [], []
    for i in range(len(hist_probs)):
        if hist_probs[i] > 0.55:
            buy_dates.append(hist_data.index[i])
            buy_prices.append(hist_data['Low'].iloc[i] * 0.98) 
        elif hist_probs[i] < 0.45:
            sell_dates.append(hist_data.index[i])
            sell_prices.append(hist_data['High'].iloc[i] * 1.02) 

    st.markdown("---")
    st.markdown(f"<h2 style='text-align: center; color: {signal_color};'>{final_signal}</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    m1, m2, m3, m4 = st.columns(4)
    current_live_price = intraday_data['Close'].iloc[-1] if intraday_data is not None and not intraday_data.empty else latest_data['Close']
    
    m1.metric("Current Live Price", f"${current_live_price:.2f}")
    m2.metric("XGBoost Probability", f"{xgb_prob * 100:.1f}% Bullish")
    m3.metric("FinBERT Sentiment", f"{sentiment_score:.2f} Score")
    m4.metric("HMM Risk Regime", "🚨 CRASH DETECTED" if current_regime == crash_state else "🟢 Stable")

    with col1:
        if intraday_data is not None and not intraday_data.empty:
            st.subheader(f"⏱️ Live Intraday Analysis ({ticker} - Today's 1-Minute Action)")
            is_up = intraday_data['Close'].iloc[-1] >= intraday_data['Open'].iloc[0]
            line_color = '#00FF00' if is_up else '#FF0000'
            
            last_min_rsi = intraday_data['RSI_1m'].iloc[-1]
            last_min_macd = intraday_data['MACD_1m'].iloc[-1]
            last_min_sig = intraday_data['MACD_Signal_1m'].iloc[-1]
            
            intra_rsi_text = "Overbought (High Risk)" if last_min_rsi > 70 else "Oversold (Bounce Potential)" if last_min_rsi < 30 else "Neutral"
            intra_macd_text = "Bullish Micro-Trend" if last_min_macd > last_min_sig else "Bearish Micro-Trend"
            
            st.info(f"**Current Minute Analysis:** The 1-minute RSI is {last_min_rsi:.1f} ({intra_rsi_text}). The 1-minute MACD shows a {intra_macd_text}.")
            
            fig_intra = go.Figure()
            fig_intra.add_trace(go.Scatter(
                x=intraday_data.index, y=intraday_data['Close'], 
                mode='lines', line=dict(color=line_color, width=2),
                fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)' if is_up else 'rgba(255, 0, 0, 0.1)', name="Live 1m Price"
            ))
            fig_intra.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), template="plotly_dark", 
                                    xaxis_title="Time (Today)", yaxis_title="Price ($)")
            st.plotly_chart(fig_intra, use_container_width=True)
            st.markdown("---")

        st.subheader(f"📊 10-Year Macro Trend & Technical Signals ({ticker})")
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            row_heights=[0.5, 0.25, 0.25], vertical_spacing=0.06,
                            subplot_titles=("Price Action & AI Decisions", "MACD & Signal Line (Trend Direction)", "RSI (Momentum)"))
        
        fig.add_trace(go.Candlestick(x=raw_data.index[-90:], open=raw_data['Open'][-90:], 
                                     high=raw_data['High'][-90:], low=raw_data['Low'][-90:], 
                                     close=raw_data['Close'][-90:], name="Daily Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index[-90:], y=processed_data['BB_High'][-90:], 
                                 line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="BB High"), row=1, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index[-90:], y=processed_data['BB_Low'][-90:], 
                                 line=dict(color='rgba(255,255,255,0.3)', dash='dot'), name="BB Low", fill='tonexty', fillcolor='rgba(255,255,255,0.05)'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                                 marker=dict(symbol='triangle-up', color='#00FF00', size=12, line=dict(color='white', width=1)), 
                                 name='AI Buy Signal'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', 
                                 marker=dict(symbol='triangle-down', color='#FF0000', size=12, line=dict(color='white', width=1)), 
                                 name='AI Sell Signal'), row=1, col=1)
        
        macd_hist = processed_data['MACD'][-90:] - processed_data['MACD_Signal'][-90:]
        colors = ['#00FF00' if val >= 0 else '#FF0000' for val in macd_hist]
        fig.add_trace(go.Bar(x=processed_data.index[-90:], y=macd_hist, marker_color=colors, name="MACD Histogram", opacity=0.5), row=2, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index[-90:], y=processed_data['MACD'][-90:], name="MACD Line", line=dict(color='#00F1FF', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=processed_data.index[-90:], y=processed_data['MACD_Signal'][-90:], name="Signal Line", line=dict(color='#FFB300', width=2)), row=2, col=1)
        
        bull_cross_idx = processed_data.index[-90:][processed_data['MACD_Bullish_Cross'][-90:]]
        bull_cross_val = processed_data['MACD'][-90:][processed_data['MACD_Bullish_Cross'][-90:]]
        bear_cross_idx = processed_data.index[-90:][processed_data['MACD_Bearish_Cross'][-90:]]
        bear_cross_val = processed_data['MACD'][-90:][processed_data['MACD_Bearish_Cross'][-90:]]
        
        fig.add_trace(go.Scatter(x=bull_cross_idx, y=bull_cross_val, mode='markers', 
                                 marker=dict(color='#00FF00', size=10, symbol='circle', line=dict(color='white', width=1)), 
                                 name="MACD Bull Cross"), row=2, col=1)
        fig.add_trace(go.Scatter(x=bear_cross_idx, y=bear_cross_val, mode='markers', 
                                 marker=dict(color='#FF0000', size=10, symbol='circle', line=dict(color='white', width=1)), 
                                 name="MACD Bear Cross"), row=2, col=1)

        fig.add_trace(go.Scatter(x=processed_data.index[-90:], y=processed_data['RSI'][-90:], name="RSI", line=dict(color='#E040FB', width=2)), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#FF0000", row=3, col=1, annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="#00FF00", row=3, col=1, annotation_text="Oversold (30)")
        
        fig.update_layout(height=800, margin=dict(l=10, r=10, t=40, b=10), template="plotly_dark", 
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)', rangeslider_visible=False) 
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Live FinBERT Sentiment Analysis")
    for headline, sentiment in zip(news, sentiment_details):
        color = "green" if sentiment['label'] == 'positive' else "red" if sentiment['label'] == 'negative' else "gray"
        st.markdown(f"- **{headline}** ➔ <span style='color:{color}'>[{sentiment['label'].upper()}: {sentiment['score']:.2f}]</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📊 Data Verification Engine & Academic Validation")
    tab1, tab2, tab3, tab4 = st.tabs(["10-Year Daily Data", "Preprocessed AI Features", "Today's 1-Minute Live Data", "🛡️ AI Model Validation Metrics"])
    
    with tab1: 
        st.write("Raw Daily OHLCV data fetched for the AI (showing latest 15 days).")
        st.dataframe(raw_data.tail(15), use_container_width=True)
    
    with tab2: 
        st.write("Engineered features calculated for the XGBoost model (showing latest 15 days).")
        st.dataframe(processed_data[['Close', 'Frac_Diff_Close', 'RSI', 'MACD', 'VIX', 'Target']].tail(15), use_container_width=True)
        
    with tab3:
        if intraday_data is not None and not intraday_data.empty:
            st.write(f"Live 1-Minute data points dynamically tracked today ({len(intraday_data)} total minutes recorded so far).")
            st.dataframe(intraday_data[['Open', 'High', 'Low', 'Close', 'Volume', 'RSI_1m', 'MACD_1m', 'MACD_Signal_1m']].tail(20), use_container_width=True)
        else:
            st.info("The US Stock Market is currently closed. 1-Minute intraday data will appear here automatically when the market opens.")
            
    with tab4:
        st.write("### 📈 XGBoost Out-of-Sample Performance (30% Test Set)")
        st.write("These metrics represent how accurately the model predicted market direction on unseen historical data.")
        
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Model Accuracy", f"{val_metrics['Accuracy'] * 100:.1f}%", help="Percentage of total correct predictions (Up or Down).")
        col_m2.metric("Precision (Bullish)", f"{val_metrics['Precision'] * 100:.1f}%", help="When the AI said the stock would go UP, how often was it right?")
        col_m3.metric("Recall (Bullish)", f"{val_metrics['Recall'] * 100:.1f}%", help="Out of all actual UP days, how many did the AI successfully catch?")
        
        st.caption("*Note: Financial markets are highly stochastic. An accuracy above 52-54% over a 10-year span is considered statistically significant edge for institutional trading.*")
