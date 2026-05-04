# Basic Earnings Call Sentiment Analyzer 

Analyzes the sentiment of earnings call transcripts and compares it against
post-earnings stock price movement. Built to answer: **does what management says predict what the stock does?**

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run analyzer.py
```

Then open `http://localhost:8501` in your browser.

## What It Does

- Scores transcript sentiment using **VADER** (a battle-tested NLP model)
- Applies **finance-specific word boosts** (e.g. "headwinds", "record revenue", "lowered guidance")
- Fetches **real stock price data** via yfinance to show the actual market reaction
- Visualizes **sentence-by-sentence sentiment flow** through the call
- Flags **bullish/bearish signal words** found in the text
- Detects **alignment or divergence** between tone and price reaction

## Where to Get Transcripts

- [Motley Fool Transcripts](https://www.fool.com/earnings-call-transcripts/) — free
- [Seeking Alpha](https://seekingalpha.com/) — freemium
- [SEC EDGAR 8-K filings](https://www.sec.gov/cgi-bin/browse-edgar) — always free

## Deploy Free on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → deploy

You'll have a live shareable link in ~2 minutes.

## Roadmap

| 1 | ✅ Built | VADER + finance boosts + price reaction |
