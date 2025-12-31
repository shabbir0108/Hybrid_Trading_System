import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="TradePro Algo Terminal", layout="wide", page_icon="📈")

# --- CUSTOM CSS FOR "PRO" LOOK ---
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    h1 { color: #4CAF50; text-align: center; }
</style>
""", unsafe_allow_html=True)

# --- 2. THE BRAIN (Backend Logic) ---
class DataEngine:
    def fetch_market_data(self, ticker, period="2y"):
        # Fetches 2 years to allow sufficient data for backtesting
        try:
            df = yf.download(ticker, period=period, progress=False)
            if df.empty: return pd.DataFrame()
            df.reset_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except: return pd.DataFrame()

    def scrape_analyst_ratings(self, ticker):
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            news_table = soup.find(id='news-table')
            
            if not news_table: return "Neutral", 0, []
            
            headlines = []
            for x in news_table.findAll('tr')[:5]:
                text = x.a.get_text()
                headlines.append(text)

            analyzer = SentimentIntensityAnalyzer()
            score = sum([analyzer.polarity_scores(h)['compound'] for h in headlines]) / len(headlines)
            sentiment = "Bullish 🚀" if score > 0.05 else "Bearish 📉" if score < -0.05 else "Neutral 😐"
            return sentiment, score, headlines
        except: return "Neutral", 0, ["Data Unavailable"]

class MLEngine:
    def prepare_data(self, df):
        if len(df) < 60: return pd.DataFrame()
        
        # Technical Indicators
        df['RSI'] = RSIIndicator(close=df["Close"], window=14).rsi()
        df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        
        # Target: 1 if Price Rises Tomorrow
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df

    def train_model(self, df):
        X = df[['RSI', 'SMA_50', 'EMA_20', 'Volume']]
        y = df['Target']
        
        # Split Data (80% Train, 20% Test)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Return model and test data for backtesting
        return model, acc, X_test, y_test

    def backtest_strategy(self, df, model):
        """
        Simulates trading $10,000 over the test period (Last 20% of data)
        """
        initial_capital = 10000
        cash = initial_capital
        position = 0
        
        # Use the test set (unseen data)
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:].copy()
        
        features = df_test[['RSI', 'SMA_50', 'EMA_20', 'Volume']]
        df_test['Predicted_Signal'] = model.predict(features)
        
        portfolio_values = []
        
        for i, row in df_test.iterrows():
            price = row['Close']
            signal = row['Predicted_Signal']
            
            # Buy Logic
            if signal == 1 and cash > price:
                shares = cash // price
                position += shares
                cash -= shares * price
            # Sell Logic
            elif signal == 0 and position > 0:
                cash += position * price
                position = 0
            
            portfolio_values.append(cash + (position * price))
            
        df_test['Portfolio_Value'] = portfolio_values
        final_value = portfolio_values[-1]
        roi = ((final_value - initial_capital) / initial_capital) * 100
        return df_test, final_value, roi

# --- 3. THE UI (FRONTEND) ---
st.title("📈 Hybrid Algorithmic Trading System")
st.markdown("### Technical Analysis + Sentiment Engine")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "NVDA").upper()
st.sidebar.markdown("---")
show_indicators = st.sidebar.checkbox("Show Moving Averages", True)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["High (Aggressive)", "Low (Conservative)"])

st.sidebar.markdown("---")
mode = st.sidebar.radio("Select Mode", ["Live Analysis Dashboard", "Testing Phase (Backtest)"])

if st.sidebar.button("🚀 Execute"):
    with st.spinner(f"Processing data for {ticker}..."):
        # Initialize
        data_engine = DataEngine()
        ml_engine = MLEngine()
        
        # Fetch
        df = data_engine.fetch_market_data(ticker)
        
        if not df.empty:
            # Analyze
            sent_label, sent_score, news = data_engine.scrape_analyst_ratings(ticker)
            df_proc = ml_engine.prepare_data(df)
            
            if not df_proc.empty:
                # Train Model
                model, acc, X_test, y_test = ml_engine.train_model(df_proc)
                
                # Get Today's Prediction
                last_row = df_proc[['RSI', 'SMA_50', 'EMA_20', 'Volume']].iloc[[-1]]
                prediction = model.predict(last_row)[0]
                algo_signal = "BUY" if prediction == 1 else "SELL"

                # --- MODE 1: LIVE DASHBOARD ---
                if mode == "Live Analysis Dashboard":
                    # Hybrid Logic
                    final_decision = algo_signal
                    reason = "Technical Model and News Agree"
                    
                    if algo_signal == "BUY" and "Bearish" in sent_label:
                        if risk_tolerance == "Low (Conservative)":
                            final_decision = "HOLD"
                            reason = "News is Negative (Risk Management)"
                    elif algo_signal == "SELL" and "Bullish" in sent_label:
                        if risk_tolerance == "Low (Conservative)":
                            final_decision = "HOLD"
                            reason = "News is Positive (Preventing Panic Sell)"

                    # Display KPIs
                    current_price = df['Close'].iloc[-1]
                    prev_price = df['Close'].iloc[-2]
                    change = current_price - prev_price
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Live Price", f"${current_price:.2f}", f"{change:.2f}")
                    c2.metric("Algo Prediction", algo_signal, f"Accuracy: {acc*100:.0f}%")
                    c3.metric("Market Sentiment", sent_label, f"Score: {sent_score:.2f}")
                    c4.metric("Final Signal", final_decision, delta_color="off")

                    if final_decision == "HOLD":
                        st.warning(f"⚠️ **System Alert:** Switched to HOLD because: {reason}")
                    else:
                        st.success(f"✅ **Confirmation:** {reason}")

                    # --- RESTORED TABS SECTION ---
                    st.markdown("---")
                    tab1, tab2, tab3 = st.tabs(["📊 Technical Chart", "📰 Live News", "🔢 Raw Data (Verification)"])

                    with tab1:
                        st.subheader("Technical Analysis Chart")
                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
                        fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                        if show_indicators:
                            fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['SMA_50'], line=dict(color='orange', width=1), name="50-Day SMA"), row=1, col=1)
                        colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
                        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)
                        fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                        st.plotly_chart(fig, use_container_width=True)

                    with tab2:
                        st.subheader("Live Analyst News")
                        for n in news:
                            st.markdown(f"> *{n}*")

                    with tab3:
                        st.subheader("Data Verification Table")
                        st.write("Below is the real-time data fetched from Yahoo Finance and the indicators calculated by our system.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**1. Raw Market Data (From yfinance)**")
                            st.dataframe(df.tail(10))
                        with col2:
                            st.write("**2. Processed Indicators (From 'ta' Lib)**")
                            st.dataframe(df_proc[['Date', 'Close', 'RSI', 'SMA_50', 'Target']].tail(10))

                # --- MODE 2: TESTING PHASE (BACKTEST) ---
                else:
                    st.subheader("🧪 Testing Phase (Backtest Results)")
                    df_bt, final_val, roi = ml_engine.backtest_strategy(df_proc, model)
                    
                    b1, b2, b3 = st.columns(3)
                    b1.metric("Start Balance", "$10,000")
                    b2.metric("Final Balance", f"${final_val:,.2f}")
                    b3.metric("Return on Investment (ROI)", f"{roi:.2f}%", delta=f"{roi:.2f}%")
                    
                    st.write("### Portfolio Growth Simulation")
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(x=df_bt['Date'], y=df_bt['Portfolio_Value'], fill='tozeroy', line=dict(color='#00CC96'), name="Portfolio Value"))
                    fig_bt.update_layout(height=400, template="plotly_dark", title="Performance on Unseen Data (Last 20%)")
                    st.plotly_chart(fig_bt, use_container_width=True)
                    
                    st.write("### Raw Trade Data")
                    st.dataframe(df_bt[['Date', 'Close', 'Predicted_Signal', 'Portfolio_Value']].tail(10))

            else:
                st.error("Not enough data for technical analysis.")
        else:
            st.error("Invalid Ticker.")
