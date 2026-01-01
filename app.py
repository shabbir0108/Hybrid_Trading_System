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

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .metric-card { background-color: #1e1e1e; border: 1px solid #333; padding: 15px; border-radius: 8px; text-align: center; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND LOGIC ---
class DataEngine:
    def fetch_market_data(self, ticker, period="2y"):
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

class MLEngine:
    def prepare_data(self, df):
        if len(df) < 60: return pd.DataFrame()
        df['RSI'] = RSIIndicator(close=df["Close"], window=14).rsi()
        df['SMA_50'] = SMAIndicator(close=df["Close"], window=50).sma_indicator()
        df['EMA_20'] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        df.dropna(inplace=True)
        return df

    def train_model(self, df):
        X = df[['RSI', 'SMA_50', 'EMA_20', 'Volume']]
        y = df['Target']
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        return model, acc

    def run_backtest(self, df, model, initial_capital=10000):
        # Only use the "Test" portion (last 20%) for simulation
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:].copy()
        
        features = df_test[['RSI', 'SMA_50', 'EMA_20', 'Volume']]
        df_test['Signal'] = model.predict(features)
        
        cash = initial_capital
        position = 0
        portfolio_values = []
        
        for i, row in df_test.iterrows():
            price = row['Close']
            if row['Signal'] == 1 and cash > price: # BUY
                shares = cash // price
                position += shares
                cash -= shares * price
            elif row['Signal'] == 0 and position > 0: # SELL
                cash += position * price
                position = 0
            portfolio_values.append(cash + (position * price))
            
        df_test['Portfolio_Value'] = portfolio_values
        roi = ((portfolio_values[-1] - initial_capital) / initial_capital) * 100
        return df_test, portfolio_values[-1], roi

# --- 3. MAIN APPLICATION UI ---
st.sidebar.title("🛠️ System Controls")
app_mode = st.sidebar.selectbox("Select System Mode", ["🔴 Live Trading Dashboard", "🧪 Backtesting Simulator"])
st.sidebar.markdown("---")

# ==========================================
# MODE 1: LIVE TRADING DASHBOARD
# ==========================================
if app_mode == "🔴 Live Trading Dashboard":
    st.title("🔴 Live Algorithmic Trading Dashboard")
    st.markdown("Real-time Technical & Sentiment Analysis")
    
    # Live Settings
    col1, col2 = st.columns([1, 3])
    with col1:
        ticker = st.text_input("Enter Ticker", "NVDA").upper()
        risk_tolerance = st.selectbox("Risk Tolerance", ["Low (Conservative)", "High (Aggressive)"])
        run_live = st.button("🚀 Analyze Live Market", type="primary")
    
    if run_live:
        data_engine = DataEngine()
        ml_engine = MLEngine()
        
        with st.spinner(f"Fetching live data for {ticker}..."):
            df = data_engine.fetch_market_data(ticker)
            if not df.empty:
                name, sector, ind = data_engine.get_company_info(ticker)
                st.subheader(f"{name} ({ticker})")
                st.caption(f"Sector: {sector} | Industry: {ind}")
                
                # Fetch Sentiment & Technicals
                sent_label, sent_score, news = data_engine.scrape_analyst_ratings(ticker)
                df_proc = ml_engine.prepare_data(df)
                
                if not df_proc.empty:
                    model, acc = ml_engine.train_model(df_proc)
                    
                    # Prediction
                    last_row = df_proc[['RSI', 'SMA_50', 'EMA_20', 'Volume']].iloc[[-1]]
                    prediction = model.predict(last_row)[0]
                    confidence = np.max(model.predict_proba(last_row))
                    algo_signal = "BUY" if prediction == 1 else "SELL"
                    
                    # Hybrid Logic
                    final_decision = algo_signal
                    reason = "Technical & Sentiment Agree"
                    if algo_signal == "BUY" and "Bearish" in sent_label and risk_tolerance.startswith("Low"):
                        final_decision = "HOLD"
                        reason = "Negative Sentiment Override"
                    elif algo_signal == "SELL" and "Bullish" in sent_label and risk_tolerance.startswith("Low"):
                        final_decision = "HOLD"
                        reason = "Positive Sentiment Override"
                    
                    # UI - KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Live Price", f"${df['Close'].iloc[-1]:.2f}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:.2f}")
                    k2.metric("Algo Accuracy", f"{acc*100:.0f}%")
                    k3.metric("Sentiment", sent_label, f"{sent_score:.2f}")
                    
                    badge_color = "#00CC96" if final_decision == "BUY" else "#EF553B" if final_decision == "SELL" else "#FFA15A"
                    with k4:
                        st.markdown(f"""<div style="background-color:{badge_color};padding:5px;border-radius:5px;text-align:center;"><h4 style="color:white;margin:0;">{final_decision}</h4></div>""", unsafe_allow_html=True)
                        st.caption(reason)
                    
                    # Chart
                    st.markdown("### 📉 Technical Chart")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume"), row=2, col=1)
                    fig.update_layout(height=500, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabs
                    t1, t2, t3 = st.tabs(["📰 Live News", "🔢 Raw Data", "🧠 Model Details"])
                    with t1:
                        for n in news: st.info(n)
                    with t2: st.dataframe(df.tail())
                    with t3: st.write(f"Model Confidence: {confidence*100:.1f}%")
                else: st.error("Not enough data.")
            else: st.error("Invalid Ticker")

# ==========================================
# MODE 2: BACKTESTING SIMULATOR
# ==========================================
elif app_mode == "🧪 Backtesting Simulator":
    st.title("🧪 Historical Backtesting Simulator")
    st.markdown("Test the strategy on past data to verify profitability.")
    
    # Backtest Settings (Separate from Live)
    with st.expander("⚙️ Simulation Settings", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            bt_ticker = st.text_input("Backtest Ticker", "TSLA").upper()
            initial_cap = st.number_input("Initial Capital ($)", 1000, 1000000, 10000)
        with col2:
            bt_period = st.selectbox("Data Period", ["1y", "2y", "5y", "10y"])
            run_bt = st.button("▶️ Run Simulation", type="primary")
            
    if run_bt:
        data_engine = DataEngine()
        ml_engine = MLEngine()
        
        with st.spinner(f"Simulating trades for {bt_ticker} over {bt_period}..."):
            df = data_engine.fetch_market_data(bt_ticker, period=bt_period)
            if not df.empty:
                df_proc = ml_engine.prepare_data(df)
                if not df_proc.empty:
                    # Train and Simulate
                    model, acc = ml_engine.train_model(df_proc)
                    df_res, final_val, roi = ml_engine.run_backtest(df_proc, model, initial_cap)
                    
                    # Results
                    st.success("Simulation Complete!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Start Balance", f"${initial_cap:,.2f}")
                    m2.metric("Final Balance", f"${final_val:,.2f}")
                    m3.metric("Total ROI", f"{roi:.2f}%", delta=f"{roi:.2f}%")
                    
                    # Performance Chart
                    st.markdown("### 📈 Portfolio Growth")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_res['Date'], y=df_res['Portfolio_Value'], fill='tozeroy', line=dict(color='#00CC96'), name="Portfolio Value"))
                    fig.update_layout(height=400, template="plotly_dark", title=f"Hypothetical Growth ({bt_ticker})")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade Log
                    st.markdown("### 📝 Trade Log (Last 10 Days)")
                    st.dataframe(df_res[['Date', 'Close', 'Signal', 'Portfolio_Value']].tail(10))
                else: st.error("Insufficient Data")
            else: st.error("Invalid Ticker")
