import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow import keras
from pathlib import Path

# ---------------- Setup ----------------
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
pio.templates.default = "plotly"

# Fixed lookback matching your LSTM snippet
LOOKBACK = 100

# Resolve app directory and model file (relative path works locally and on Streamlit Cloud)
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "Stock Prediction Model.keras"

# ---------- Sidebar ----------
st.sidebar.subheader("📊 Stock Settings")
popular = {
    "Apple (AAPL)": "AAPL", "Microsoft (MSFT)": "MSFT",
    "Alphabet (GOOGL)": "GOOGL", "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA", "Meta (META)": "META", "NVIDIA (NVDA)": "NVDA"
}
pick = st.sidebar.selectbox("Pick a ticker symbol", ["Custom"] + list(popular.keys()))
symbol = st.sidebar.text_input("Or enter symbol", value="GOOG").upper()
if pick != "Custom":
    symbol = popular[pick]

today = dt.date.today()
start = st.sidebar.date_input("Start date", value=today.replace(year=today.year - 5), max_value=today)


run = st.sidebar.button("Run", use_container_width=True)

# ---------- Utilities ----------
def load_data(tkr: str, start_date: dt.date, end_date: dt.date) -> tuple[pd.DataFrame, dict]:
    df = yf.download(
        tickers=tkr, start=start_date, end=end_date, interval="1d",
        auto_adjust=True, progress=False, multi_level_index=False
    )
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if df is None or df.empty:
        return pd.DataFrame(), {}
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"]).reset_index(drop=True)
    try:
        info = yf.Ticker(tkr).info
    except Exception:
        info = {}
    return df, info

def model_rating_from_r2(r2: float) -> str:
    if r2 >= 0.85: return "Excellent"
    if r2 >= 0.70: return "Good"
    if r2 >= 0.50: return "Fair"
    return "Poor"

def confidence_badge_from_r2(r2: float) -> tuple[str, str]:
    if r2 >= 0.75: return ("High", "#16a34a")
    if r2 >= 0.50: return ("Medium", "#f59e0b")
    return ("Low", "#ef4444")

# ---------- Main ----------
st.markdown('<h1 style="text-align:center">📈 Stock Price Predictor</h1>', unsafe_allow_html=True)
st.caption("Stock Price Prediction powered by LSTM Neural Networks – capturing trends beyond simple averages.")

if not run:
    st.info("Pick a ticker, set dates, and click Run.")
    st.stop()

with st.spinner(f"Loading {symbol}..."):
    df, info = load_data(symbol, start, today)

if df.empty:
    st.error("No data returned for this symbol/date range.")
    st.stop()

# Top metrics
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
chg = last_close - prev_close
chg_pct = (chg / prev_close * 100) if prev_close else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Company", info.get("longName", symbol))
c2.metric("Current Price", f"${last_close:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
c3.metric("Data Points", len(df))
c4.metric("Sector", info.get("sector", "N/A"))

st.write("")
st.write("")
st.write("")

# ---------- Moving Averages row (3 charts side-by-side) ----------
st.subheader("Moving Averages")

# Compute SMAs (use min_periods=window to start lines only after enough data)
ma = df[["Date", "Close"]].copy()
ma["MA50"]  = ma["Close"].rolling(window=50,  min_periods=50).mean()
ma["MA100"] = ma["Close"].rolling(window=100, min_periods=100).mean()
ma["MA200"] = ma["Close"].rolling(window=200, min_periods=200).mean()

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.caption("Price vs MA50")
    fig_ma50 = go.Figure()
    fig_ma50.add_trace(go.Scatter(x=ma["Date"], y=ma["Close"], name="Close",
                                  mode="lines", line=dict(color="#00D4FF", width=2)))
    fig_ma50.add_trace(go.Scatter(x=ma["Date"], y=ma["MA50"], name="MA50",
                                  mode="lines", line=dict(color="#F59E0B", width=2.2)))
    fig_ma50.update_layout(height=320, hovermode="x unified",
                           xaxis_title=None, yaxis_title="Price (USD)",
                           xaxis_rangeslider_visible=False, template="plotly_dark",
                           margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ma50, use_container_width=True)

with col2:
    st.caption("Price vs MA50 vs MA100")
    fig_ma100 = go.Figure()
    fig_ma100.add_trace(go.Scatter(x=ma["Date"], y=ma["Close"], name="Close",
                                   mode="lines", line=dict(color="#00D4FF", width=2)))
    fig_ma100.add_trace(go.Scatter(x=ma["Date"], y=ma["MA50"], name="MA50",
                                   mode="lines", line=dict(color="#F59E0B", width=2.0, dash="dot")))
    fig_ma100.add_trace(go.Scatter(x=ma["Date"], y=ma["MA100"], name="MA100",
                                   mode="lines", line=dict(color="#FF6B6B", width=2.2)))
    fig_ma100.update_layout(height=320, hovermode="x unified",
                            xaxis_title=None, yaxis_title="Price (USD)",
                            xaxis_rangeslider_visible=False, template="plotly_dark",
                            margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ma100, use_container_width=True)

with col3:
    st.caption("Price vs MA100 vs MA200")
    fig_ma200 = go.Figure()
    fig_ma200.add_trace(go.Scatter(x=ma["Date"], y=ma["Close"], name="Close",
                                   mode="lines", line=dict(color="#00D4FF", width=2)))
    fig_ma200.add_trace(go.Scatter(x=ma["Date"], y=ma["MA100"], name="MA100",
                                   mode="lines", line=dict(color="#FF6B6B", width=2.0, dash="dot")))
    fig_ma200.add_trace(go.Scatter(x=ma["Date"], y=ma["MA200"], name="MA200",
                                   mode="lines", line=dict(color="#16A34A", width=2.2)))
    fig_ma200.update_layout(height=320, hovermode="x unified",
                            xaxis_title=None, yaxis_title="Price (USD)",
                            xaxis_rangeslider_visible=False, template="plotly_dark",
                            margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ma200, use_container_width=True)

st.write("")
st.write("")
st.write("")

# Price history (line only)
left, right = st.columns([1, 24])
with left:
    st.image("images/history.png")
with right:
    st.subheader("Price History")
try:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], mode="lines",
        name="Close", line=dict(color="#00D4FF", width=2.6)
    ))
    fig.update_layout(
        height=420, hovermode="x unified", xaxis_title=None, yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False, title=f"{symbol} Close"
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.warning("Could not render price chart. Showing fallback.")
    st.line_chart(df.set_index("Date")["Close"])

# Ensure enough rows for the 80/20 split with 100-day context
min_needed = LOOKBACK + 6
if len(df) <= min_needed:
    st.warning(f"Not enough data (need > {min_needed} rows). Try a longer date range.")
    st.stop()

# ---------- LSTM pipeline (correct scaling and inverse transform) ----------
# Split close into train/test by 80/20 on raw index
close = df["Close"].astype(float).reset_index(drop=True)
split_idx = int(len(close) * 0.80)

data_train = pd.DataFrame(close.iloc[:split_idx], columns=["Close"])
data_test = pd.DataFrame(close.iloc[split_idx:],  columns=["Close"])

# Fit scaler ONLY on training data to avoid leakage
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data_train[["Close"]])

# Transform the tail of train (for context) and the test part
train_tail_scaled = scaler.transform(data_train.tail(LOOKBACK)[["Close"]])  # (LOOKBACK, 1)
test_scaled = scaler.transform(data_test[["Close"]])                  # (n_test, 1)

# Concatenate to build the rolling window over test
data_test_scale = np.vstack([train_tail_scaled, test_scaled])               # (LOOKBACK + n_test, 1)

# ---------- Sequence building for test (fixed) ----------
X_test_seq, y_test_seq = [], []

# Use the first dimension of shape (number of rows)
for i in range(LOOKBACK, data_test_scale.shape[0]):
    # window is (LOOKBACK, 1)
    X_test_seq.append(data_test_scale[i - LOOKBACK:i, :])
    # single target value (scaled)
    y_test_seq.append(data_test_scale[i, 0])

# Convert to arrays with correct ranks
X_test_seq = np.asarray(X_test_seq, dtype=np.float32)        # (n, LOOKBACK, 1)
if X_test_seq.ndim == 2:                                     # safety for (n, LOOKBACK)
    X_test_seq = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[10], 1)

y_test_seq = np.asarray(y_test_seq, dtype=np.float32).reshape(-1, 1)  # (n, 1)


# Load the saved LSTM model and predict on prepared sequences
with st.spinner("Loading LSTM model and running inference..."):
    kmodel = keras.models.load_model(MODEL_PATH)
    yhat_scaled = kmodel.predict(X_test_seq, verbose=0)     # (n, 1)

# Invert scaling to price units using the scaler's inverse_transform
pred_prices = scaler.inverse_transform(yhat_scaled).ravel()  # (n,)
true_prices = scaler.inverse_transform(y_test_seq).ravel()   # (n,)

# Align dates for this test segment: one date per test sample
test_dates = df["Date"].iloc[split_idx:].reset_index(drop=True)

# Metrics (R²/MAE) on the aligned arrays
if len(true_prices) == len(pred_prices):
    r2 = float(r2_score(true_prices, pred_prices))
    mae = float(mean_absolute_error(true_prices, pred_prices))
else:
    r2, mae = 0.0, float("nan")

rating = model_rating_from_r2(r2)

m1, m2, m3 = st.columns(3)
m1.metric("Test R² Score", f"{r2:.3f}")
m2.metric("Test MAE", f"${mae:.2f}")
m3.metric("Model Rating", rating)

st.write("")
st.write("")
st.write("")
# Actual vs Predicted chart (your LSTM outputs vs true)
left, right = st.columns([1,24])
with left:
    st.image("images/compare-image.png")
with right:
    st.subheader("Predictions vs Actual")
try:
    figp = go.Figure()
    figp.add_trace(go.Scatter(x=test_dates, y=true_prices, mode="lines", name="Actual",
                              line=dict(color="#00D4FF", width=2.2)))
    figp.add_trace(go.Scatter(x=test_dates, y=pred_prices, mode="lines", name="Predicted",
                              line=dict(color="#FF6B6B", width=2.2, dash="dash")))
    figp.update_layout(height=420, hovermode="x unified",
                       xaxis_title=None, yaxis_title="Price (USD)",
                       title="Actual vs Predicted Prices (Using LSTM)")
    st.plotly_chart(figp, use_container_width=True)
except Exception:
    st.warning("Could not render prediction chart.")

st.write("")
st.write("")
st.write("")
# Next‑day prediction (single step) from the last LOOKBACK scaled window
left, right = st.columns([1,24])
with left:
    st.image("images/future.png")
with right:
    st.subheader("Next Day Prediction")

last_window_scaled = data_test_scale[-LOOKBACK:, :].reshape(1, LOOKBACK, 1)  # (1, LOOKBACK, 1)
next_scaled = kmodel.predict(last_window_scaled, verbose=0)                  # (1, 1)
next_price = float(scaler.inverse_transform(next_scaled)[0, 0])              # proper inverse

delta = next_price - last_close
c1, c2, c3 = st.columns(3)
c1.metric("Current Close", f"${last_close:,.2f}")
c2.metric("Predicted Next Close", f"${next_price:,.2f}", f"{delta:+.2f}")
c3.metric("Lookback (days)", LOOKBACK)

# Confidence banner from test R²
conf_text, conf_color = confidence_badge_from_r2(r2)
st.markdown(
    f"""
    <div style="
        margin-top:0.5rem;padding:0.9rem 1rem;border-radius:10px;
        background:#0f172a;border:1px solid #334155;">
        <span style="font-size:1.05rem;">💡 Prediction Confidence:
            <b style="color:{conf_color};">{conf_text}</b>
            <span style="opacity:0.8;">(based on test R² of {r2:.3f})</span>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")
st.write("")
st.write("")
# Volume (enhanced)
if "Volume" in df.columns and df["Volume"].notna().any():
    left, right = st.columns([1,24])
    with left:
        st.image("images/volumes.png")
    with right:
        st.subheader("Volumes Traded")
    try:
        vol = df[["Date", "Volume"]].copy()
        vol["Vol_MA7"] = vol["Volume"].rolling(7, min_periods=1).mean()

        fig_vol = go.Figure()
        fig_vol.add_trace(
            go.Scatter(
                x=vol["Date"], y=vol["Volume"], mode="lines",
                line=dict(color="rgba(255,165,0,0.15)", width=0),
                fill="tozeroy", name="Volume area",
                hoverinfo="skip", showlegend=False
            )
        )
        fig_vol.add_trace(
            go.Bar(
                x=vol["Date"], y=vol["Volume"], name="Volume",
                marker=dict(color="rgba(255,165,0,0.9)", line=dict(color="rgba(255,165,0,1.0)", width=0.6))
            )
        )
        fig_vol.add_trace(
            go.Scatter(
                x=vol["Date"], y=vol["Vol_MA7"], mode="lines", name="7D Vol SMA",
                line=dict(color="#FF6B6B", width=3)
            )
        )
        fig_vol.update_layout(
            height=360, hovermode="x unified", xaxis_title=None, yaxis_title="Volume",
            bargap=0.15, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_rangeslider_visible=False, template="plotly_dark"
        )
        fig_vol.update_yaxes(tickformat="~s")
        st.plotly_chart(fig_vol, use_container_width=True)
    except Exception:
        st.warning("Could not render volume chart.")

# Footer
st.markdown("---")
st.caption("Dataset from Yahoo Finance API. ")
st.caption("<div style='text-align:center'>© 2025 Ayush Behera</div>", unsafe_allow_html=True)
