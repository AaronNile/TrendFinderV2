#!/usr/bin/env python3
"""
TrendFinder Flask app
"""
from flask import Flask, request, render_template_string, jsonify
import yfinance as fin
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
import pandas as pd
import io
import base64
import math
import numpy as np

app = Flask(__name__)

VOL_SCALE = 0.2


def simpleMA(closing_list):
    # This will be the long-term MA (span = 50)
    span = 50
    i = 0
    simpleMA = []

    while i < len(closing_list):
        if i < span - 1:
            simpleMA.append(None)
            i += 1
            continue

        # Creating a sliced list of N elements, where N is the span, then finding the mean value
        slicedList = closing_list[i - span + 1:i + 1]
        average = sum(slicedList) / span
        simpleMA.append(average)
        i += 1

    return simpleMA


def ExponentialMA(closing_list):
    # 12-day EMA
    if not closing_list:
        return []

    span = 12
    # The smoothing factor is given by 2/(span+1)
    smoothing = 2 / (span + 1)

    ExponentialMA = [closing_list[0]]
    i = 1

    while i < len(closing_list):
        # Formula: (P * A) + (EV * (1-A))
        EMA_val = (closing_list[i] * smoothing) + (ExponentialMA[i - 1] * (1 - smoothing))
        ExponentialMA.append(EMA_val)
        i += 1

    return ExponentialMA


def Volatility(closing_list):
    # Annualized volatility computed from daily returns (sample std * sqrt(252))
    returns_list = []
    i = 1

    while i < len(closing_list):
        # finding returns for each day's closing price
        prev = closing_list[i - 1]
        if prev == 0:
            i += 1
            continue

        val = (closing_list[i] - prev) / prev
        returns_list.append(val)
        i += 1

    if len(returns_list) < 2:
        return 0.0

    returns_avg = sum(returns_list) / len(returns_list)

    variance = sum((j - returns_avg) ** 2 for j in returns_list) / (len(returns_list) - 1)
    standard_dev = variance ** 0.5
    volatility = standard_dev * (252 ** 0.5)  # annualize

    return volatility


def find_trend_text(SMA, EMA, Vol, ticker):
    # return the textual trend assessment

    # Find last non-None SMA and EMA values safely
    try:
        sma_last = next(x for x in reversed(SMA) if x is not None)
    except StopIteration:
        return "Not enough SMA data to determine trend."

    try:
        ema_last = next(x for x in reversed(EMA) if x is not None)
    except StopIteration:
        return "Not enough EMA data to determine trend."

    Upperbound = sma_last + Vol
    Lowerbound = sma_last - Vol

    if ema_last < sma_last:
        if ema_last < Lowerbound:
            return f"{ticker}: Exponential moving average below the Simple Moving Average and the lower bound, a strong sign of a downtrend."
        else:
            return f"{ticker}: EMA below SMA but above the lower bound, a weak sign for a downtrend; {ticker} may be in a consolidation phase."
    else:
        if ema_last > Upperbound:
            return f"{ticker}: EMA above SMA and above the upper bound, a strong sign of an uptrend."
        else:
            return f"{ticker}: EMA above SMA but below the upper bound, a weak sign for an uptrend; {ticker} may be in a consolidation phase."


def create_lower_bound(SMA, Vol):
    lower_bound = []
    for s, v in zip(SMA, Vol):
        if s is not None and v is not None:
            lower_bound.append(s - v)
        else:
            lower_bound.append(None)
    return lower_bound


def create_upper_bound(SMA, Vol):
    upper_bound = []
    for s, v in zip(SMA, Vol):
        if s is not None and v is not None:
            upper_bound.append(s + v)
        else:
            upper_bound.append(None)
    return upper_bound


def plot_to_base64(dates, SMA, EMA, lower_bound, upper_bound, ticker):
    # Create a PNG plot and return a base64-encoded string

    sma_arr = np.array([np.nan if x is None else float(x) for x in SMA])
    ema_arr = np.array([np.nan if x is None else float(x) for x in EMA])
    lower_arr = np.array([np.nan if x is None else float(x) for x in lower_bound])
    upper_arr = np.array([np.nan if x is None else float(x) for x in upper_bound])

    fig = plot.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1)

    # Convert dates (pandas.DatetimeIndex or list) to a plottable form
    ax.plot(dates, sma_arr, label='SMA', linestyle='-', linewidth=2)
    ax.plot(dates, ema_arr, label='EMA', linestyle='-', linewidth=2)
    ax.plot(dates, lower_arr, label='Lower Bound', linestyle='--', linewidth=0.8)
    ax.plot(dates, upper_arr, label='Upper Bound', linestyle='--', linewidth=0.8)
    ax.fill_between(
        dates,
        lower_arr,
        upper_arr,
        where=~np.isnan(lower_arr) & ~np.isnan(upper_arr),
        alpha=0.08
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{ticker} Trend")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate(rotation=25)

    bio = io.BytesIO()
    fig.savefig(bio, format='png', bbox_inches='tight', dpi=150)
    plot.close(fig)
    bio.seek(0)

    b64 = base64.b64encode(bio.read()).decode('ascii')
    return b64


# Flask routes

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>TrendFinder</title>
  <style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f7fb;
    color: #222;
    max-width: 960px;
    margin: 0 auto;
    padding: 2rem 1rem;
  }
  h1 {
    text-align: center;
    margin-bottom: 0.25rem;
  }
  .subtitle {
    text-align: center;
    color: #555;
    margin-bottom: 2rem;
  }
  .card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    margin-bottom: 2rem;
  }
  form {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
  }
  input[type=text] {
    padding: 0.6rem;
    font-size: 1rem;
    border-radius: 8px;
    border: 1px solid #ccc;
    width: 180px;
  }
  button {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    border: none;
    background: #4f46e5;
    color: white;
    font-size: 1rem;
    cursor: pointer;
  }
  button:hover {
    background: #4338ca;
  }
  .result h3 {
    margin-bottom: 0.3rem;
  }
  .trend {
    background: #eef2ff;
    border-left: 4px solid #4f46e5;
    padding: 0.75rem;
    border-radius: 6px;
    margin-bottom: 1rem;
  }
  img {
    max-width: 100%;
    border-radius: 8px;
    border: 1px solid #ddd;
  }
  .error {
    color: #b91c1c;
    text-align: center;
  }
  </style>
</head>
<body>
  <h1>TrendFinder</h1>
  <p class="subtitle">Analyze medium-term stock trends using moving averages and volatility bands.</p>

  <div class="card">
    <form method="POST" action="/">
      <label for="ticker">Ticker:</label>
      <input id="ticker" name="ticker" type="text" placeholder="AAPL" value="{{ ticker or '' }}" oninput="this.value=this.value.toUpperCase()" required />
      <button type="submit">Analyze</button>
    </form>
  </div>

  {% if error %}
    <p class="error">{{ error }}</p>
  {% endif %}

  {% if trend %}
    <div class="card result">
      <h3>{{ ticker }}</h3>
      <div class="trend">{{ trend }}</div>
      <p><em>Note: data from Yahoo Finance via yfinance; plot generated server-side.</em></p>
      <img src="data:image/png;base64,{{ image_b64 }}" alt="Trend chart" />
    </div>
  {% endif %}

  <hr/>
  <p style="font-size:0.85rem;color:#666">
    Tip: If the app returns "no data", try another ticker or wait a minute and retry (rate limits possible).
  </p>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template_string(INDEX_HTML)

    # process ticker
    ticker = request.form.get("ticker", "").strip().upper()
    if not ticker:
        return render_template_string(INDEX_HTML, error="Please enter a ticker symbol.")

    # Build date range: last year
    date_today = pd.to_datetime("today").normalize()
    date_year_ago = date_today - pd.DateOffset(years=1)
    start = date_year_ago.strftime("%Y-%m-%d")
    end = date_today.strftime("%Y-%m-%d")

    # Download historical price data
    try:
        df = fin.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        return render_template_string(INDEX_HTML, error=f"Error fetching data: {str(e)}")

    if df is None or df.empty:
        return render_template_string(INDEX_HTML, error=f"No data returned for {ticker}. Please check the symbol.")

    # Extract closing prices and dates
    closing_list = df.iloc[:, 0].tolist()
    dates = df.index.to_list()

    # Compute moving averages and volatility
    EMA = ExponentialMA(closing_list)
    SMA = simpleMA(closing_list)
    Vol_scalar = Volatility(closing_list) * VOL_SCALE

    # Build price-proportional volatility per-SMA-point
    price_vol = [(s * Vol_scalar) if s is not None else None for s in SMA]

    # Build upper and lower bounds
    lower_bound = create_lower_bound(SMA, price_vol)
    upper_bound = create_upper_bound(SMA, price_vol)

    trend_text = find_trend_text(SMA, EMA, Vol_scalar, ticker)

    image_b64 = plot_to_base64(dates, SMA, EMA, lower_bound, upper_bound, ticker)

    return render_template_string(
        INDEX_HTML,
        ticker=ticker,
        trend=trend_text,
        image_b64=image_b64
    )


@app.route("/api/trend")
def api_trend():
    ticker = request.args.get("ticker", "").strip().upper()
    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    date_today = pd.to_datetime("today").normalize()
    date_year_ago = date_today - pd.DateOffset(years=1)
    start = date_year_ago.strftime("%Y-%m-%d")
    end = date_today.strftime("%Y-%m-%d")

    try:
        df = fin.download(ticker, start=start, end=end, progress=False)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if df is None or df.empty:
        return jsonify({"error": "no data returned for ticker"}), 404

    closing_list = df.iloc[:, 0].tolist()
    dates = [d.strftime("%Y-%m-%d") for d in df.index]

    EMA = ExponentialMA(closing_list)
    SMA = simpleMA(closing_list)
    Vol_scalar = Volatility(closing_list) * VOL_SCALE

    price_vol = [(s * Vol_scalar) if s is not None else None for s in SMA]
    lower_bound = create_lower_bound(SMA, price_vol)
    upper_bound = create_upper_bound(SMA, price_vol)

    trend_text = find_trend_text(SMA, EMA, Vol_scalar, ticker)

    def to_simple(arr):
        return [(float(x) if x is not None else None) for x in arr]

    return jsonify({
        "ticker": ticker,
        "dates": dates,
        "close": to_simple(df.iloc[:, 0].tolist()),
        "sma": to_simple(SMA),
        "ema": to_simple(EMA),
        "lower": to_simple(lower_bound),
        "upper": to_simple(upper_bound),
        "vol": float(Vol_scalar),
        "trend_text": trend_text
    })


if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=5000, debug=True)
