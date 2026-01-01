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

# --- CUSTOM CSS FOR PROFESSIONAL UI ---
st.markdown("""
<style>
    /* Dark Theme Cards */
    .metric-card {
        background-color: #1e1e1e;
        border: 1px solid #333;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Green/Red text for signals */
    .buy-signal { color: #00CC96; font-weight: bold; font-size: 24px; }
    .sell-signal { color: #EF553B; font-weight: bold; font-size: 24px; }
    .hold-signal { color: #FFA15A; font-weight: bold; font-size: 24px; }
    
    /* Headers */
    h1, h2, h3 { font-family: 'Roboto', sans-serif; }
    
    /* Divider */
    hr { margin-top: 10px; margin-bottom: 10px; border: 0; border-top: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 2. BACKEND ENGINE ---
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
        """Fetches Company Name & Sector for the UI"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('longName', ticker), info.get('sector', 'Unknown'), info.get('industry', 'Unknown')
        except:
            return ticker, "N/A", "N/A"

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
        return model, acc, X_test, y_test

    def backtest_strategy(self, df, model):
        initial_capital = 10000
        cash = initial_capital
        position = 0
        split = int(len(df) * 0.8)
        df_test = df.iloc[split:].copy()
        
        features = df_test[['RSI', 'SMA_50', 'EMA_20', 'Volume']]
        df_test['Predicted_Signal'] = model.predict(features)
        
        portfolio_values = []
        for i, row in df_test.iterrows():
            price = row['Close']
            signal = row['Predicted_Signal']
            if signal == 1 and cash > price:
                shares = cash // price
                position += shares
                cash -= shares * price
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

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker = st.sidebar.text_input("Ticker Symbol", "NVDA").upper()
st.sidebar.caption("Examples: AAPL, GOOGL, TSLA, RELIANCE.NS")
st.sidebar.markdown("---")
risk_tolerance = st.sidebar.select_slider("Risk Tolerance", options=["Low (Conservative)", "High (Aggressive)"])
show_indicators = st.sidebar.toggle("Show Moving Averages", True)
mode = st.sidebar.radio("System Mode", ["Live Analysis Dashboard", "Testing Phase (Backtest)"])

if st.sidebar.button("🚀 Initialize System", type="primary"):
    data_engine = DataEngine()
    ml_engine = MLEngine()

    with st.status("System Status", expanded=True) as status:
        st.write("📡 Connecting to Market Data Feed...")
        df = data_engine.fetch_market_data(ticker)
        
        if not df.empty:
            st.write(f"🏢 Fetching Profile for {ticker}...")
            name, sector, industry = data_engine.get_company_info(ticker)
            
            st.write("📰 Scraping Analyst Ratings...")
            sent_label, sent_score, news = data_engine.scrape_analyst_ratings(ticker)
            
            st.write("🧠 Training Algorithmic Models...")
            df_proc = ml_engine.prepare_data(df)
            
            if not df_proc.empty:
                model, acc, X_test, y_test = ml_engine.train_model(df_proc)
                status.update(label="Analysis Complete!", state="complete", expanded=False)
                
                # --- COMPANY HEADER ---
                st.markdown(f"## {name}")
                st.markdown(f"**Sector:** {sector} | **Industry:** {industry}")
                st.markdown("---")

                # Live Prediction Logic
                last_row = df_proc[['RSI', 'SMA_50', 'EMA_20', 'Volume']].iloc[[-1]]
                prediction = model.predict(last_row)[0]
                confidence = np.max(model.predict_proba(last_row)) # Confidence score
                algo_signal = "BUY" if prediction == 1 else "SELL"

                # Hybrid Decision Logic
                final_decision = algo_signal
                reason = "Technical & Sentiment Agree"
                if algo_signal == "BUY" and "Bearish" in sent_label and risk_tolerance.startswith("Low"):
                    final_decision = "HOLD"
                    reason = "Negative Sentiment Override"
                elif algo_signal == "SELL" and "Bullish" in sent_label and risk_tolerance.startswith("Low"):
                    final_decision = "HOLD"
                    reason = "Positive Sentiment Override"

                # --- MODE 1: LIVE DASHBOARD ---
                if mode == "Live Analysis Dashboard":
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}", f"{df['Close'].iloc[-1]-df['Close'].iloc[-2]:.2f}")
                    
                    with col2:
                        st.metric("Algo Accuracy", f"{acc*100:.0f}%")
                        st.progress(confidence, text=f"AI Confidence: {confidence*100:.0f}%")
                        
                    with col3:
                        st.metric("Sentiment Score", f"{sent_score:.2f}", sent_label)
                        
                    with col4:
                        color = "green" if final_decision == "BUY" else "red" if final_decision == "SELL" else "orange"
                        st.markdown(f":{color}-background-co[**{final_decision}**]")
                        st.caption(f"Reason: {reason}")

                    # Charts
                    st.subheader("📉 Technical Chart")
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
                    fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                    if show_indicators:
                        fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['SMA_50'], line=dict(color='orange', width=1), name="SMA 50"), row=1, col=1)
                    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker_color='rgba(100, 200, 255, 0.6)'), row=2, col=1)
                    fig.update_layout(height=600, template="plotly_dark", xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabs
                    tab1, tab2, tab3 = st.tabs(["📰 Scraped News", "🔢 Data Verification", "📥 Download"])
                    with tab1:
                        for n in news: st.info(f"📰 {n}")
                    with tab2:
                        st.dataframe(df_proc.tail(10))
                    with tab3:
                        csv = df_proc.to_csv().encode('utf-8')
                        st.download_button("Download Analysis Data (CSV)", csv, "trading_data.csv", "text/csv")

                # --- MODE 2: TESTING PHASE ---
                else:
                    st.subheader("🧪 Backtesting Simulation")
                    df_bt, final_val, roi = ml_engine.backtest_strategy(df_proc, model)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Initial Investment", "$10,000")
                    m2.metric("Final Portfolio Value", f"${final_val:,.2f}")
                    m3.metric("Total Return (ROI)", f"{roi:.1f}%", delta=f"{roi:.1f}%")
                    
                    st.area_chart(df_bt.set_index("Date")["Portfolio_Value"], color="#00CC96")
                    st.dataframe(df_bt[['Date', 'Close', 'Predicted_Signal', 'Portfolio_Value']].tail())

            else: st.error("Insufficient Data for Analysis")
        else: st.error("Ticker Not Found")
