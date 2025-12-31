import streamlit as st
import yfinance as yf
import pandas as pd
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
    def fetch_market_data(self, ticker, period="1y"):
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
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Prediction
        prediction = model.predict(X.iloc[[-1]])[0]
        return ("BUY" if prediction == 1 else "SELL"), acc

# --- 3. THE UI (FRONTEND) ---
st.title("📈 Hybrid Algorithmic Trading System")
st.markdown("### Final Year Project | Technical Analysis + Sentiment Engine")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "NVDA").upper()
st.sidebar.markdown("---")
show_indicators = st.sidebar.checkbox("Show Moving Averages", True)
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["High (Aggressive)", "Low (Conservative)"])

if st.sidebar.button("🚀 Analyze Market"):
    with st.spinner(f"Connecting to live markets for {ticker}..."):
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
                algo_signal, acc = ml_engine.train_model(df_proc)
                
                # --- HYBRID LOGIC ENGINE ---
                final_decision = algo_signal
                reason = "Technical Model and News Agree"
                
                # Conflict Handling
                if algo_signal == "BUY" and "Bearish" in sent_label:
                    if risk_tolerance == "Low (Conservative)":
                        final_decision = "HOLD"
                        reason = "News is Negative (Risk Management)"
                elif algo_signal == "SELL" and "Bullish" in sent_label:
                    if risk_tolerance == "Low (Conservative)":
                        final_decision = "HOLD"
                        reason = "News is Positive (Preventing Panic Sell)"

                # --- DASHBOARD ROW 1: KPI CARDS ---
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

                # --- DASHBOARD ROW 2: ADVANCED CHART ---
                st.subheader("📊 Technical Analysis Chart")
                
                # Create Subplots (Price on top, Volume on bottom)
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # Candlestick
                fig.add_trace(go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'], name="Price"), row=1, col=1)
                
                # Indicators (If checked in sidebar)
                if show_indicators:
                    fig.add_trace(go.Scatter(x=df_proc['Date'], y=df_proc['SMA_50'], 
                                             line=dict(color='orange', width=1), name="50-Day SMA"), row=1, col=1)
                
                # Volume
                colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
                fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=colors, name="Volume"), row=2, col=1)

                fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)

                # --- DASHBOARD ROW 3: DATA TABS ---
                st.markdown("---")
                tab1, tab2 = st.tabs(["📰 Live News Feed", "🔢 Raw Data Verification"])
                
                with tab1:
                    for n in news:
                        st.markdown(f"> *{n}*")
                
                with tab2:
                    st.dataframe(df_proc.tail(10).style.highlight_max(axis=0))

            else:
                st.error("Not enough data for technical analysis.")
        else:
            st.error("Invalid Ticker.")
