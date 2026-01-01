import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- LIBRARIES FOR ENSEMBLE ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TradePro Ensemble Terminal", layout="wide", page_icon="🧠")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND: DATA ENGINE ---
class DataEngine:
    def fetch_market_data(self, ticker, period="5y"): 
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty: return pd.DataFrame()
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except: return pd.DataFrame()

    def get_company_info(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('longName', ticker), info.get('sector', 'N/A'), info.get('industry', 'N/A')
        except: return ticker, "N/A", "N/A"

    def scrape_analyst_ratings(self, ticker):
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            news_table = soup.find(id='news-table')
            if not news_table: return "Neutral", 0, []
            headlines = [x.a.get_text() for x in news_table.findAll('tr')[:5]]
            analyzer = SentimentIntensityAnalyzer()
            score = sum([analyzer.polarity_scores(h)['compound'] for h in headlines]) / len(headlines)
            sentiment = "Bullish 🚀" if score > 0.05 else "Bearish 📉" if score < -0.05 else "Neutral 😐"
            return sentiment, score, headlines
        except: return "Neutral", 0, ["Data Unavailable"]

# --- 3. BACKEND: DUAL MODEL ENGINE ---
class ModelEngine:
    def prepare_data(self, df):
        if len(df) < 100: return pd.DataFrame()
        # Common Indicators
        df['RSI'] = RSIIndicator(close=df["Close"], window=14).rsi()
        df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df

    # --- ML: RANDOM FOREST ---
    def run_random_forest(self, df):
        X = df[['RSI', 'SMA_50', 'EMA_20', 'Volume']]
        y = df['Target']
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Test Metrics
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Live Prediction
        last_row = X.iloc[[-1]]
        pred_signal = model.predict(last_row)[0]
        
        return pred_signal, acc, X_test, model

    # --- DL: LSTM ---
    def run_lstm(self, df):
        data = df[['Close', 'RSI', 'SMA_50', 'EMA_20']].values
        target = df['Target'].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        lookback = 60
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(scaled_data[i-lookback:i])
            y.append(target[i])
        X, y = np.array(X), np.array(y)
        
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Build Network
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 4)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=0)
        
        # Test Metrics
        preds_prob = model.predict(X_test, verbose=0)
        preds = (preds_prob > 0.5).astype(int)
        acc = accuracy_score(y_test, preds)
        
        # Live Prediction
        last_60 = scaled_data[-lookback:]
        X_input = np.array([last_60])
        pred_prob = model.predict(X_input, verbose=0)[0][0]
        pred_signal = 1 if pred_prob > 0.5 else 0
        
        return pred_signal, acc, X_test, model

    def backtest(self, df, signals, initial_capital=10000):
        # Generic Backtester
        cash = initial_capital
        position = 0
        portfolio = []
        prices = df['Close'].values[-len(signals):] # Align lengths
        
        for i in range(len(signals)):
            price = prices[i]
            signal = signals[i]
            if signal == 1 and cash > price:
                shares = cash // price
                position += shares
                cash -= shares * price
            elif signal == 0 and position > 0:
                cash += position * price
                position = 0
            portfolio.append(cash + (position * price))
        
        if len(portfolio) == 0: return [initial_capital], 0
        roi = ((portfolio[-1] - initial_capital) / initial_capital) * 100
        return portfolio, roi

# --- 4. MAIN UI ---
st.sidebar.title("🛠️ System Controls")
app_mode = st.sidebar.selectbox("System Mode", ["🔴 Live Ensemble Analysis", "🧪 Comparative Backtesting"])
st.sidebar.markdown("---")
st.sidebar.info("System Architecture: **Hybrid Ensemble (RF + LSTM)**")

if app_mode == "🔴 Live Ensemble Analysis":
    st.title("🔴 Live Ensemble Trading Dashboard")
    st.markdown("Combines **Machine Learning (RF)** and **Deep Learning (LSTM)** for superior decision making.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Ticker", "NVDA").upper()
        risk = st.selectbox("Risk", ["Low", "High"])
        btn = st.button("🚀 Analyze Market", type="primary")
        
    if btn:
        data = DataEngine()
        engine = ModelEngine()
        
        with st.spinner(f"Training Dual-Engine Models for {ticker}..."):
            df = data.fetch_market_data(ticker)
            if not df.empty and len(df) > 200:
                name, sec, ind = data.get_company_info(ticker)
                st.subheader(f"{name} ({ticker})")
                
                # 1. Sentiment
                s_label, s_score, news = data.scrape_analyst_ratings(ticker)
                
                # 2. Data Prep
                df_proc = engine.prepare_data(df)
                
                # 3. RUN BOTH MODELS SIMULTANEOUSLY
                rf_sig, rf_acc, _, _ = engine.run_random_forest(df_proc)
                lstm_sig, lstm_acc, _, _ = engine.run_lstm(df_proc)
                
                # 4. ENSEMBLE LOGIC (The "Meta-Vote")
                # Weights: Sentiment is a veto, ML and DL vote.
                
                final_decision = "HOLD"
                reason = "Models Disagree (Uncertainty)"
                
                # Case A: Both Models Agree
                if rf_sig == 1 and lstm_sig == 1:
                    final_decision = "BUY"
                    reason = "Strong Buy (ML + DL Agree)"
                elif rf_sig == 0 and lstm_sig == 0:
                    final_decision = "SELL"
                    reason = "Strong Sell (ML + DL Agree)"
                # Case B: Disagreement
                else:
                    # If models disagree, default to HOLD or follow higher accuracy model
                    if rf_acc > lstm_acc:
                        final_decision = "BUY" if rf_sig == 1 else "SELL"
                        reason = f"Models Diverge (Followed ML - Higher Acc: {rf_acc*100:.0f}%)"
                    else:
                        final_decision = "BUY" if lstm_sig == 1 else "SELL"
                        reason = f"Models Diverge (Followed DL - Higher Acc: {lstm_acc*100:.0f}%)"
                
                # Case C: Sentiment Override (Risk Management)
                if final_decision == "BUY" and "Bearish" in s_label and risk == "Low":
                    final_decision = "HOLD"
                    reason = "Risk Alert: Negative News Override"
                
                # 5. UI DISPLAY
                # Top Row: Models
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Live Price", f"${df['Close'].iloc[-1]:.2f}")
                c2.metric("ML (Random Forest)", "BUY" if rf_sig==1 else "SELL", f"Acc: {rf_acc*100:.0f}%")
                c3.metric("DL (LSTM Network)", "BUY" if lstm_sig==1 else "SELL", f"Acc: {lstm_acc*100:.0f}%")
                
                # Final Decision Badge
                color = "#00CC96" if final_decision == "BUY" else "#EF553B" if final_decision == "SELL" else "#FFA15A"
                c4.markdown(f"""<div style="background-color:{color};padding:5px;border-radius:5px;text-align:center;color:white;"><b>{final_decision}</b></div>""", unsafe_allow_html=True)
                st.caption(f"Logic: {reason}")
                
                st.divider()
                
                # Charts
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume']), row=2, col=1)
                fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("📰 Live News Analysis"):
                    st.write(f"**Overall Sentiment:** {s_label} (Score: {s_score:.2f})")
                    for n in news: st.info(n)

            else: st.error("Invalid Ticker or Insufficient Data")

elif app_mode == "🧪 Comparative Backtesting":
    st.title("🧪 Comparative Backtesting (ML vs DL)")
    st.markdown("Prove which model performs better on historical data.")
    
    bt_ticker = st.text_input("Backtest Ticker", "TSLA").upper()
    
    if st.button("▶️ Run Championship Simulation"):
        data = DataEngine()
        engine = ModelEngine()
        
        with st.spinner("Training both models on history..."):
            df = data.fetch_market_data(bt_ticker)
            if not df.empty:
                df_proc = engine.prepare_data(df)
                
                # 1. Backtest RF
                _, rf_acc, X_test_rf, model_rf = engine.run_random_forest(df_proc)
                rf_signals = model_rf.predict(X_test_rf)
                rf_port, rf_roi = engine.backtest(df_proc, rf_signals)
                
                # 2. Backtest LSTM
                _, lstm_acc, X_test_lstm, model_lstm = engine.run_lstm(df_proc)
                # Bulk predict
                preds_prob = model_lstm.predict(X_test_lstm, verbose=0)
                lstm_signals = (preds_prob > 0.5).astype(int).flatten()
                lstm_port, lstm_roi = engine.backtest(df_proc, lstm_signals)
                
                # Display Winner
                st.success("Simulation Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🤖 Random Forest (ML)")
                    st.metric("Total Return (ROI)", f"{rf_roi:.2f}%")
                    st.metric("Accuracy", f"{rf_acc*100:.1f}%")
                
                with col2:
                    st.subheader("🧠 LSTM Network (DL)")
                    st.metric("Total Return (ROI)", f"{lstm_roi:.2f}%")
                    st.metric("Accuracy", f"{lstm_acc*100:.1f}%")
                
                # Comparison Chart
                st.subheader("📈 Performance Comparison Chart")
                chart_df = pd.DataFrame({
                    "Date": df_proc['Date'].values[-len(rf_port):],
                    "ML (Random Forest)": rf_port,
                    "DL (LSTM)": lstm_port
                }).set_index("Date")
                
                st.line_chart(chart_df)
                
            else: st.error("Error fetching data")
