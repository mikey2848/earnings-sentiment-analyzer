"""
Earnings Call Sentiment Analyzer - Layer 1
========================================
Analyzes sentiment from earnings call transcripts and compares
against post-earnings stock price movement.

Setup:
    pip install vaderSentiment yfinance plotly streamlit requests beautifulsoup4 pandas

Usage:
    streamlit run analyzer.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
import re

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Earnings Sentiment Analyzer",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        background-color: #0d0d0d;
        color: #e8e8e8;
    }
    .main { background-color: #0d0d0d; }
    .stTextArea textarea {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        background-color: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #333;
        border-radius: 6px;
    }
    .stTextInput input {
        background-color: #1a1a1a;
        color: #e8e8e8;
        border: 1px solid #333;
        border-radius: 6px;
    }
    .metric-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Mono', monospace;
        font-size: 2.2rem;
        font-weight: 500;
        margin: 4px 0;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #888;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .positive { color: #4ade80; }
    .negative { color: #f87171; }
    .neutral  { color: #94a3b8; }
    h1 { font-family: 'DM Sans', sans-serif; font-weight: 600; letter-spacing: -0.02em; }
    h3 { font-family: 'DM Sans', sans-serif; font-weight: 400; color: #aaa; }
    .stButton button {
        background: #18181b;
        color: #e8e8e8;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        padding: 8px 20px;
        transition: all 0.15s;
    }
    .stButton button:hover {
        background: #27272a;
        border-color: #71717a;
    }
    .insight-box {
        background: #111827;
        border-left: 3px solid #4ade80;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #d1d5db;
    }
    .warning-box {
        background: #111827;
        border-left: 3px solid #fbbf24;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.9rem;
        color: #d1d5db;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SENTIMENT ENGINE
# ─────────────────────────────────────────────

analyzer = SentimentIntensityAnalyzer()

# Finance-specific word boosts (VADER misses these)
BULLISH_WORDS = {
    "record": 1.5, "beat": 1.5, "exceeded": 1.5, "raised": 1.2, "raised guidance": 2.0,
    "accelerating": 1.3, "outperformed": 1.5, "momentum": 1.0, "confidence": 1.0,
    "growth": 0.8, "expanding": 1.0, "strong demand": 1.5, "robust": 1.2,
    "margin expansion": 1.5, "share gains": 1.2, "raised outlook": 2.0,
}

BEARISH_WORDS = {
    "headwinds": -1.5, "uncertainty": -1.2, "challenging": -1.3, "softness": -1.2,
    "missed": -1.5, "lowered guidance": -2.0, "lowered outlook": -2.0, "cautious": -1.2,
    "macro pressures": -1.5, "slower": -1.0, "deceleration": -1.5, "below expectations": -2.0,
    "cost pressures": -1.2, "inventory": -0.5, "writedown": -1.5, "restructuring": -1.0,
}


def get_finance_adjusted_score(text: str) -> dict:
    """Run VADER + apply finance-specific word boosts."""
    base = analyzer.polarity_scores(text)
    compound = base["compound"]

    text_lower = text.lower()
    boost = 0.0
    triggered = []

    for word, weight in BULLISH_WORDS.items():
        if word in text_lower:
            boost += weight * 0.05
            triggered.append((word, weight))

    for word, weight in BEARISH_WORDS.items():
        if word in text_lower:
            boost += weight * 0.05
            triggered.append((word, weight))

    adjusted = max(-1.0, min(1.0, compound + boost))

    return {
        "compound_raw": round(compound, 4),
        "compound_adjusted": round(adjusted, 4),
        "positive": round(base["pos"], 4),
        "negative": round(base["neg"], 4),
        "neutral": round(base["neu"], 4),
        "finance_boost": round(boost, 4),
        "triggered_words": triggered,
    }


def analyze_by_sentence(text: str) -> pd.DataFrame:
    """Score each sentence individually for the sentiment timeline chart."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    rows = []
    for i, sentence in enumerate(sentences):
        scores = analyzer.polarity_scores(sentence)
        rows.append({
            "index": i,
            "sentence": sentence[:120] + ("..." if len(sentence) > 120 else ""),
            "compound": scores["compound"],
            "label": "Positive" if scores["compound"] >= 0.05
                     else ("Negative" if scores["compound"] <= -0.05 else "Neutral")
        })
    return pd.DataFrame(rows)


def get_price_reaction(ticker: str, earnings_date: str) -> dict:
    """Get stock return from close before earnings to close/current price after.
    Handles today's date by falling back to intraday 1m data for current price."""
    try:
        dt = datetime.strptime(earnings_date, "%Y-%m-%d")
        today = datetime.now().date()
        is_today = dt.date() == today

        stock = yf.Ticker(ticker)

        # Get price BEFORE earnings (always use daily close)
        hist_before = stock.history(start=dt - timedelta(days=7), end=dt + timedelta(days=1))
        if hist_before.empty:
            return {"error": "No price data found before that date. Check the ticker."}

        hist_before.index = hist_before.index.tz_localize(None)
        hist_before = hist_before.sort_index()
        price_before = float(hist_before["Close"].iloc[-1])
        date_before  = str(hist_before.index[-1].date())

        # Get price AFTER earnings
        if is_today:
            intraday = stock.history(period="1d", interval="1m")
            if intraday.empty:
                return {"error": "Market may not be open yet or ticker is invalid."}
            intraday.index = intraday.index.tz_localize(None)
            price_after = float(intraday["Close"].iloc[-1])
            date_after  = str(intraday.index[-1].strftime("%Y-%m-%d %H:%M"))
            note = "live intraday price"
        else:
            hist_after = stock.history(start=dt, end=dt + timedelta(days=7))
            if hist_after.empty:
                return {"error": "No price data found after that date."}
            hist_after.index = hist_after.index.tz_localize(None)
            hist_after = hist_after.sort_index()
            price_after = float(hist_after["Close"].iloc[0])
            date_after  = str(hist_after.index[0].date())
            note = "closing price"

        pct_change = ((price_after - price_before) / price_before) * 100

        return {
            "ticker": ticker.upper(),
            "date_before": date_before,
            "date_after":  date_after,
            "price_before": round(price_before, 2),
            "price_after":  round(price_after, 2),
            "pct_change":   round(pct_change, 2),
            "note": note,
        }
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# SAMPLE TRANSCRIPTS (for quick testing)
# ─────────────────────────────────────────────

SAMPLES = {
    "Apple Q1 2024 (Strong Beat)": {
        "ticker": "AAPL",
        "date": "2024-02-01",
        "text": """
We are pleased to report record revenue for the December quarter, with iPhone revenue growing strongly
across all geographies. Services hit an all-time high and we are incredibly confident in our pipeline.
Our installed base of active devices reached a new record. We saw robust demand across all product
categories and strong momentum in emerging markets. We are raising our outlook for the next quarter
and expanding our share buyback program. Customer satisfaction remains extraordinarily high and we
continue to gain market share in smartphones globally. Our gross margins expanded to 45.9 percent,
above our guidance range. We remain very confident in our long-term trajectory.
""".strip()
    },
    "Generic Weak Quarter": {
        "ticker": "META",
        "date": "2022-10-27",
        "text": """
We are facing significant headwinds in the advertising market due to macroeconomic uncertainty
and increased competition. Revenue missed expectations and we are seeing softness in demand across
several verticals. We remain cautious about the near term outlook given macro pressures. Our expenses
increased substantially due to investments in the metaverse. We are lowering our guidance for Q4
and expect challenging conditions to persist into next year. Headcount growth will be meaningfully
slower and we are reviewing our cost structure. There is uncertainty around the pace of recovery.
Inventory adjustments are ongoing at many of our advertising partners.
""".strip()
    },
}


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────

st.markdown("## 📊 Earnings Call Sentiment Analyzer")
st.markdown("### *Does what management says predict what the stock does?*")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### ⚡ Quick Load")
    sample_choice = st.selectbox("Load a sample transcript", ["(none)"] + list(SAMPLES.keys()))
    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
**Layer 1** uses VADER sentiment analysis with custom finance-specific word boosts.

1. Paste a transcript (or load a sample)
2. Enter the ticker + earnings date
3. See sentiment score vs. stock reaction

**Where to get transcripts:**
- [Motley Fool](https://fool.com/earnings-call-transcripts/)
- [Seeking Alpha](https://seekingalpha.com/)
- [SEC EDGAR 8-K filings](https://www.sec.gov/cgi-bin/browse-edgar)
""")
    st.markdown("---")
    st.markdown("**Layer 2** → Section-level analysis (coming soon)")
    st.markdown("**Layer 3** → LLM-powered signals (coming soon)")
    st.markdown("**Layer 4** → Backtesting across 50+ calls (coming soon)")

# Main inputs
col1, col2 = st.columns([2, 1])

with col1:
    default_text = SAMPLES[sample_choice]["text"] if sample_choice != "(none)" else ""
    transcript = st.text_area(
        "Paste Earnings Call Transcript",
        value=default_text,
        height=280,
        placeholder="Paste the earnings call transcript here. You can use prepared remarks, Q&A, or the full transcript..."
    )

with col2:
    default_ticker = SAMPLES[sample_choice]["ticker"] if sample_choice != "(none)" else ""
    default_date   = SAMPLES[sample_choice]["date"]   if sample_choice != "(none)" else ""

    ticker = st.text_input("Ticker Symbol", value=default_ticker, placeholder="e.g. AAPL")
    earnings_date = st.text_input("Earnings Date (YYYY-MM-DD)", value=default_date, placeholder="e.g. 2024-02-01")
    company_name  = st.text_input("Company Name (optional)", placeholder="e.g. Apple Inc.")

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔍 Analyze", use_container_width=True)


# ─────────────────────────────────────────────
# ANALYSIS OUTPUT
# ─────────────────────────────────────────────

if run:
    if not transcript.strip():
        st.error("Please paste a transcript first.")
    else:
        with st.spinner("Analyzing sentiment..."):

            scores      = get_finance_adjusted_score(transcript)
            sentence_df = analyze_by_sentence(transcript)
            price_data  = get_price_reaction(ticker, earnings_date) if ticker and earnings_date else None

        st.markdown("---")
        title = company_name if company_name else ticker.upper() if ticker else "Earnings Call"
        st.markdown(f"### Results — {title}")

        # ── Top metrics ──
        adj    = scores["compound_adjusted"]
        color  = "positive" if adj >= 0.05 else ("negative" if adj <= -0.05 else "neutral")
        label  = "Bullish 🟢" if adj >= 0.05 else ("Bearish 🔴" if adj <= -0.05 else "Neutral ⚪")

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentiment Score</div>
                <div class="metric-value {color}">{adj:+.3f}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">% Positive Sentences</div>
                <div class="metric-value positive">{scores['positive']*100:.1f}%</div>
                <div class="metric-label">of all content</div>
            </div>""", unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">% Negative Sentences</div>
                <div class="metric-value negative">{scores['negative']*100:.1f}%</div>
                <div class="metric-label">of all content</div>
            </div>""", unsafe_allow_html=True)

        with m4:
            if price_data and "error" not in price_data:
                pct    = price_data["pct_change"]
                pcol   = "positive" if pct >= 0 else "negative"
                plab   = f"{price_data['date_before']} → {price_data['date_after']}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stock Reaction</div>
                    <div class="metric-value {pcol}">{pct:+.2f}%</div>
                    <div class="metric-label">{plab}</div>
                </div>""", unsafe_allow_html=True)
            else:
                msg = price_data["error"] if price_data else "Enter ticker + date"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Stock Reaction</div>
                    <div class="metric-value neutral">—</div>
                    <div class="metric-label" style="color:#ef4444">{msg[:40]}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sentiment timeline chart ──
        st.markdown("#### Sentiment Flow Through the Call")

        fig = go.Figure()
        colors_map = {"Positive": "#4ade80", "Negative": "#f87171", "Neutral": "#475569"}

        fig.add_trace(go.Bar(
            x=sentence_df["index"],
            y=sentence_df["compound"],
            marker_color=[colors_map[l] for l in sentence_df["label"]],
            hovertext=sentence_df["sentence"],
            hovertemplate="<b>Sentence %{x}</b><br>Score: %{y:.3f}<br>%{hovertext}<extra></extra>",
            name="Sentence Sentiment"
        ))

        # Rolling average
        window = max(3, len(sentence_df) // 10)
        sentence_df["rolling"] = sentence_df["compound"].rolling(window, center=True).mean()
        fig.add_trace(go.Scatter(
            x=sentence_df["index"],
            y=sentence_df["rolling"],
            line=dict(color="#facc15", width=2),
            name=f"{window}-sentence trend"
        ))

        fig.add_hline(y=0, line_dash="dot", line_color="#555", line_width=1)

        fig.update_layout(
            paper_bgcolor="#0d0d0d",
            plot_bgcolor="#111111",
            font=dict(family="DM Mono", color="#aaa"),
            xaxis=dict(title="Sentence #", gridcolor="#1f1f1f", showgrid=True),
            yaxis=dict(title="Sentiment Score", gridcolor="#1f1f1f", showgrid=True,
                       range=[-1.1, 1.1]),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#333"),
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Finance word triggers ──
        if scores["triggered_words"]:
            st.markdown("#### Finance Signal Words Detected")
            tw_df = pd.DataFrame(scores["triggered_words"], columns=["Word / Phrase", "Weight"])
            tw_df = tw_df.sort_values("Weight", ascending=False)

            fcol1, fcol2 = st.columns(2)
            bullish_hits = tw_df[tw_df["Weight"] > 0]
            bearish_hits = tw_df[tw_df["Weight"] < 0]

            with fcol1:
                if not bullish_hits.empty:
                    st.markdown("**🟢 Bullish signals**")
                    for _, row in bullish_hits.iterrows():
                        st.markdown(f"- `{row['Word / Phrase']}` (+{row['Weight']:.1f})")

            with fcol2:
                if not bearish_hits.empty:
                    st.markdown("**🔴 Bearish signals**")
                    for _, row in bearish_hits.iterrows():
                        st.markdown(f"- `{row['Word / Phrase']}` ({row['Weight']:.1f})")

        # ── Insight box ──
        st.markdown("<br>", unsafe_allow_html=True)
        if price_data and "error" not in price_data:
            pct = price_data["pct_change"]
            sentiment_dir = "bullish" if adj >= 0.05 else ("bearish" if adj <= -0.05 else "neutral")
            price_dir = "up" if pct >= 0 else "down"
            aligned = (adj >= 0.05 and pct >= 0) or (adj <= -0.05 and pct < 0)

            if aligned:
                st.markdown(f"""
                <div class="insight-box">
                ✅ <strong>Aligned:</strong> The transcript read as <strong>{sentiment_dir}</strong>
                and the stock moved <strong>{price_dir} {abs(pct):.2f}%</strong> — sentiment and price
                reaction pointed in the same direction.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                ⚠️ <strong>Divergence:</strong> The transcript read as <strong>{sentiment_dir}</strong>
                but the stock moved <strong>{price_dir} {abs(pct):.2f}%</strong>. This could mean
                guidance or numbers mattered more than tone — worth digging into.
                </div>""", unsafe_allow_html=True)

        # ── Raw scores expander ──
        with st.expander("Raw Scores"):
            st.json({
                "compound_raw":      scores["compound_raw"],
                "compound_adjusted": scores["compound_adjusted"],
                "finance_boost":     scores["finance_boost"],
                "positive_ratio":    scores["positive"],
                "negative_ratio":    scores["negative"],
                "neutral_ratio":     scores["neutral"],
                "sentences_analyzed": len(sentence_df),
                "price_data": price_data,
            })

        st.markdown("---")
        st.markdown("""
        <div style='color:#555; font-size:0.78rem; font-family: DM Mono, monospace;'>
        Layer 1 · VADER + Finance Boosts · yfinance price data · Not financial advice
        </div>
        """, unsafe_allow_html=True)
