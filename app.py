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
st.set_page_config(page_title="TradePro Ensemble", layout="wide", page_icon="📈")

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
            sentiment = "Bullish" if score > 0.05 else "Bearish" if score < -0.05 else "Neutral"
            return sentiment, score, headlines
        except: return "Neutral", 0, ["Data Unavailable"]

# --- 3. BACKEND: DUAL MODEL ENGINE ---
class ModelEngine:
    def prepare_data(self, df):
        if len(df) < 100: return pd.DataFrame()
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
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(lookback, 4)))
        model.add(Dropout(0.2))
        model.add(LSTM(50))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, batch_size=32, epochs=3, verbose=0)
        
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
        cash = initial_capital
        position = 0
        portfolio = []
        prices = df['Close'].values[-len(signals):] 
        
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
st.sidebar.title("System Controls")
app_mode = st.sidebar.selectbox("Choose Mode", ["Live Ensemble Analysis", "Comparative Backtesting"])
st.sidebar.info("Running: Random Forest (ML) + LSTM (DL)")

if app_mode == "Live Ensemble Analysis":
    st.title("Live Ensemble Trading Dashboard")
    st.write("Real-time analysis using Machine Learning & Deep Learning.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        ticker = st.text_input("Enter Ticker", "NVDA").upper()
        risk = st.selectbox("Risk Tolerance", ["Low", "High"])
        if st.button("Analyze Now", type="primary"):
            data = DataEngine()
            engine = ModelEngine()
            
            with st.spinner("Processing ML & DL Models..."):
                df = data.fetch_market_data(ticker)
                if not df.empty and len(df) > 200:
                    name, sec, ind = data.get_company_info(ticker)
                    
                    # 1. Models
                    df_proc = engine.prepare_data(df)
                    rf_sig, rf_acc, _, _ = engine.run_random_forest(df_proc)
                    lstm_sig, lstm_acc, _, _ = engine.run_lstm(df_proc)
                    
                    # 2. Sentiment
                    s_label, s_score, news = data.scrape_analyst_ratings(ticker)
                    
                    # 3. Logic
                    final = "HOLD"
                    reason = "Uncertainty (Models Disagree)"
                    
                    if rf_sig == 1 and lstm_sig == 1:
                        final = "BUY"; reason = "Strong Signal (Models Agree)"
                    elif rf_sig == 0 and lstm_sig == 0:
                        final = "SELL"; reason = "Strong Signal (Models Agree)"
                    else:
                         # Disagreement Logic
                        if rf_acc > lstm_acc:
                            final = "BUY" if rf_sig == 1 else "SELL"
                            reason = "Models Diverge (Trusting ML Accuracy)"
                        else:
                            final = "BUY" if lstm_sig == 1 else "SELL"
                            reason = "Models Diverge (Trusting DL Accuracy)"

                    # Risk Override
                    if final == "BUY" and s_label == "Bearish" and risk == "Low":
                        final = "HOLD"; reason = "Negative News Override"

                    # 4. Display (Clean Native UI)
                    st.header(f"{name} ({ticker})")
                    st.markdown("---")
                    
                    # Row 1: The Result
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
                    c2.metric("Sentiment", f"{s_label}", f"{s_score:.2f}")
                    
                    with c3:
                        st.write("### Final Signal")
                        if final == "BUY":
                            st.success(f"**BUY** ({reason})")
                        elif final == "SELL":
                            st.error(f"**SELL** ({reason})")
                        else:
                            st.warning(f"**HOLD** ({reason})")

                    # Row 2: Model Details
                    st.markdown("#### Model Performance")
                    m1, m2 = st.columns(2)
                    m1.info(f"**Random Forest (ML):** {'BUY' if rf_sig==1 else 'SELL'} (Acc: {rf_acc*100:.0f}%)")
                    m2.info(f"**LSTM Network (DL):** {'BUY' if lstm_sig==1 else 'SELL'} (Acc: {lstm_acc*100:.0f}%)")

                    # Row 3: Charts
                    st.markdown("---")
                    st.subheader("Technical Chart")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']), row=1, col=1)
                    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume']), row=2, col=1)
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("Latest News Headlines"):
                        for n in news: st.text(f"• {n}")
                else:
                    st.error("Ticker not found or insufficient data.")

elif app_mode == "Comparative Backtesting":
    st.title("Comparative Backtesting (ML vs DL)")
    bt_ticker = st.text_input("Enter Ticker for Simulation", "TSLA").upper()
    
    if st.button("Run Simulation", type="primary"):
        data = DataEngine()
        engine = ModelEngine()
        
        with st.spinner("Simulating..."):
            df = data.fetch_market_data(bt_ticker)
            if not df.empty and len(df) > 100:
                df_proc = engine.prepare_data(df)
                
                # ML Backtest
                _, rf_acc, X_test_rf, model_rf = engine.run_random_forest(df_proc)
                rf_signals = model_rf.predict(X_test_rf)
                rf_port, rf_roi = engine.backtest(df_proc, rf_signals)
                
                # DL Backtest
                _, lstm_acc, X_test_lstm, model_lstm = engine.run_lstm(df_proc)
                preds_prob = model_lstm.predict(X_test_lstm, verbose=0)
                lstm_signals = (preds_prob > 0.5).astype(int).flatten()
                lstm_port, lstm_roi = engine.backtest(df_proc, lstm_signals)
                
                # Fix Arrays (Slicing)
                min_len = min(len(rf_port), len(lstm_port))
                rf_port = rf_port[-min_len:]
                lstm_port = lstm_port[-min_len:]
                dates = df_proc['Date'].values[-min_len:]
                
                # Results
                st.success("Simulation Complete")
                c1, c2 = st.columns(2)
                c1.metric("ML (Random Forest) ROI", f"{rf_roi:.2f}%")
                c2.metric("DL (LSTM) ROI", f"{lstm_roi:.2f}%")
                
                chart_df = pd.DataFrame({"ML": rf_port, "DL": lstm_port}, index=dates)
                st.line_chart(chart_df)
            else:
                st.error("Error fetching data.")
