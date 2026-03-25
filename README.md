# Trader Performance vs Market Sentiment
### Primetrade.ai — Data Science Internship Assignment

---

## Objective

Analyze how Bitcoin market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid. Uncover patterns that could inform smarter trading strategies.

---

## Project Structure
```
primetrade_assignment/
├── data/
│   ├── fear_greed_index.csv       # Bitcoin Fear/Greed Index (2018–2025)
│   └── historical_data.csv        # Hyperliquid trader history (2023–2025)
├── notebooks/
│   └── analysis.ipynb             # Main analysis notebook
├── charts/
│   ├── chart1_performance_by_sentiment.png
│   ├── chart2_behavior_by_sentiment.png
│   ├── chart3_segment_performance.png
│   ├── chart4_heatmap_sentiment_segment.png
│   ├── chart5_trader_archetypes.png
│   └── chart6_timeline_sentiment_pnl.png
└── README.md
```

---

## Setup & How to Run

### Requirements
```
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter plotly openpyxl
```

### Run the Notebook
```
cd primetrade_assignment/notebooks
jupyter notebook analysis.ipynb
```

Run all cells in order (Kernel → Restart & Run All).

---

## Methodology

### Data Preparation
- Loaded and documented both datasets (shape, dtypes, nulls, duplicates)
- Parsed timestamps and aligned datasets at daily granularity
- Engineered key metrics: daily PnL, win rate, leverage proxy, long/short ratio, trade frequency
- Simplified sentiment into binary (Fear / Greed) for cleaner analysis

### Analysis
- Compared PnL, win rate, and trade behavior across Fear vs Greed days
- Segmented traders into 3 dimensions: leverage, frequency, consistency
- Built heatmap showing Sentiment × Segment interaction
- Trained a Gradient Boosting model to predict next-day profitability
- Clustered traders into 4 behavioral archetypes using KMeans

---

## Key Insights

| # | Insight |
|---|---|
| 1 | Greed days produce **2x higher median PnL** ($243 vs $123) |
| 2 | Traders use **higher leverage on Fear days** (4.39 vs 3.10) — increasing risk during volatility |
| 3 | **High leverage traders thrive on Fear days** — low leverage traders dominate on Greed days |
| 4 | Only **9 of 32 traders are consistent winners** — discipline beats aggression |
| 5 | **Win rate (0.298) and leverage (0.270)** are top predictors of next-day profitability |
| 6 | **4 behavioral archetypes** identified: Hyperactive, Consistent Performer, Balanced, Underperformer |

---

## Strategy Recommendations

**Strategy 1 — Sentiment-Based Leverage Adjustment**
- Fear days → High leverage traders stay active (3x better PnL than on Greed days)
- Greed days → Reduce leverage, ride the trend steadily

**Strategy 2 — Consistency Over Big Wins**
- Maintain win rate above 40% with controlled position sizing
- During Fear days, reduce frequency but maintain discipline
- Consistent winners profit on both Fear ($303) and Greed ($697) days

---

## Bonus

- **Predictive Model:** Gradient Boosting Classifier achieves **66.45% accuracy** predicting next-day profitability
- **Trader Clustering:** KMeans identifies 4 behavioral archetypes with distinct risk/reward profiles

---

## Author

Submitted as part of the Data Science Internship application at **Primetrade.ai**