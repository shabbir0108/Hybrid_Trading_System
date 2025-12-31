import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION (MUST BE FIRST) ---
st.set_page_config(page_title="Hybrid AI Trader", layout="wide")

# --- 2. THE "BRAIN" (CLASSES) ---
class DataEngine:
    def fetch_market_data(self, ticker, period="1y"):
        try:
            # Force download to avoid cache issues
            df = yf.download(ticker, period=period, progress=False)
            if df.empty:
                return pd.DataFrame()
            df.reset_index(inplace=True)
            # Flatten multi-level columns if present (common yfinance issue)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    def scrape_analyst_ratings(self, ticker):
        """Scrapes news/ratings to satisfy the 'Scraping' requirement"""
        try:
            # Use Finviz for analyst ratings (simpler to scrape)
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the news table
            news_table = soup.find(id='news-table')
            if not news_table:
                return "Neutral (No Data)", 0, ["No recent news found."]

            headlines = []
            tr_rows = news_table.findAll('tr')
            for index, row in enumerate(tr_rows):
                text = row.a.get_text()
                headlines.append(text)
                if index == 4: break # Limit to top 5
            
            # Sentiment Analysis
            analyzer = SentimentIntensityAnalyzer()
            total_score = 0
            for h in headlines:
                score = analyzer.polarity_scores(h)['compound']
                total_score += score
            
            avg_score = total_score / len(headlines) if headlines else 0
            sentiment = "Bullish 🟢" if avg_score > 0.05 else "Bearish 🔴" if avg_score < -0.05 else "Neutral ⚪"
            
            return sentiment, avg_score, headlines
        except Exception as e:
            return "Neutral (Error)", 0, [f"Scraping blocked: {str(e)}"]

class MLEngine:
    def prepare_data(self, df):
        if df.empty or len(df) < 60:
            return pd.DataFrame()
            
        # Calculate Indicators using 'ta' library
        # RSI
        rsi_ind = RSIIndicator(close=df["Close"], window=14)
        df['RSI'] = rsi_ind.rsi()
        
        # SMA
        sma_ind = SMAIndicator(close=df["Close"], window=50)
        df['SMA_50'] = sma_ind.sma_indicator()
        
        # Target: 1 if Close Price tomorrow > Close Price today
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        df.dropna(inplace=True)
        return df

    def train_model(self, df):
        features = ['RSI', 'SMA_50', 'Open', 'Volume']
        X = df[features]
        y = df['Target']
        
        # Split & Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Accuracy
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        # Predict Tomorrow
        last_row = X.iloc[[-1]]
        prediction = model.predict(last_row)[0]
        signal = "BUY 🟢" if prediction == 1 else "SELL 🔴"
        
        return signal, acc

# --- 3. THE UI (STREAMLIT) ---
st.title("📈 Hybrid Algorithmic Trading System ")
st.markdown("Combines **Live Market Data**, **Web Scraping**, and **Machine Learning**.")

# Sidebar Input
ticker_input = st.sidebar.text_input("Enter Stock Ticker:", "NVDA")
st.sidebar.info("Try: AAPL, TSLA, MSFT, RELIANCE.NS")

if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Analyzing {ticker_input}..."):
        # Initialize Engines
        data_engine = DataEngine()
        ml_engine = MLEngine()
        
        # 1. Fetch Data
        df = data_engine.fetch_market_data(ticker_input)
        
        if df.empty:
            st.error("Could not fetch data. Check ticker symbol or internet connection.")
        else:
            # 2. Scrape Sentiment
            sentiment_label, sentiment_score, news_headlines = data_engine.scrape_analyst_ratings(ticker_input)
            
            # 3. AI Prediction
            df_processed = ml_engine.prepare_data(df)
            if not df_processed.empty:
                ai_signal, accuracy = ml_engine.train_model(df_processed)
                
                # 4. HYBRID LOGIC (The "Better Project" feature)
                final_decision = ai_signal
                # Conflict Check: If AI says BUY but News is Bad -> HOLD
                if "BUY" in ai_signal and "Bearish" in sentiment_label:
                    final_decision = "HOLD 🟠 (Risk Alert)"
                elif "SELL" in ai_signal and "Bullish" in sentiment_label:
                    final_decision = "HOLD 🟠 (News Conflict)"
                
                # --- DISPLAY RESULTS ---
                # Top Metrics
                col1, col2, col3, col4 = st.columns(4)
                current_price = df['Close'].iloc[-1]
                col1.metric("Price", f"${current_price:.2f}")
                col2.metric("AI Signal", ai_signal)
                col3.metric("Sentiment", sentiment_label)
                col4.metric("Final Call", final_decision)
                
                st.write(f"**Model Accuracy:** {accuracy*100:.1f}%")
                
                # Chart
                st.subheader("Technical Analysis")
                fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'])])
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # News
                st.subheader("Scraped Analyst News")
                for news in news_headlines:
                    st.text(f"📰 {news}")
            else:
                st.warning("Not enough data to calculate indicators (Need > 60 days).")
            
            # --- VERIFICATION SECTION (Now indented correctly) ---
            st.markdown("---")
            st.subheader("🔎 Behind the Scenes (Data Verification)")
            
            # Tab layout for cleaner look
            tab1, tab2, tab3 = st.tabs(["📊 Raw Market Data", "🧠 Technical Indicators", "📰 Scraped News"])
            
            with tab1:
                st.write("**Source:** Yahoo Finance (Live)")
                st.dataframe(df.tail(5))
                
            with tab2:
                st.write("**Source:** Calculated via 'ta' library")
                st.dataframe(df_processed[['Date', 'Close', 'RSI', 'SMA_50', 'Target']].tail(5))
                
            with tab3:
                st.write("**Source:** Finviz.com (Scraped)")
                if news_headlines:
                    for i, news in enumerate(news_headlines):
                        st.write(f"**{i+1}.** {news}")
                else:
                    st.write("No news found.")