import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(
    page_title="Trader Sentiment Analysis | Primetrade.ai",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252840);
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #00d4ff;
        margin: 8px 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4ff; }
    .metric-label { font-size: 0.85rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; }
    .insight-box {
        background: linear-gradient(135deg, #1a1f35, #1e2440);
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #f39c12;
        margin: 10px 0;
        color: #e0e6f0;
    }
    .strategy-box {
        background: linear-gradient(135deg, #1a2f1a, #1e3820);
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #2ecc71;
        margin: 10px 0;
        color: #e0e6f0;
    }
    .header-gradient {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
    div[data-testid="stTabs"] button {
        font-size: 0.95rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# header
st.markdown('<p class="header-gradient">📈 Trader Performance vs Market Sentiment</p>', unsafe_allow_html=True)
st.markdown("**Primetrade.ai — Data Science Internship Assignment** | *Rushikesh Baban Kedar*")
st.markdown("---")

# load data
@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentiment_path = os.path.join(base, "data", "fear_greed_index.csv")
    trades_path    = os.path.join(base, "data", "historical_data.csv")

    sentiment = pd.read_csv(sentiment_path)
    trades    = pd.read_csv(trades_path)

    sentiment['date'] = pd.to_datetime(sentiment['date'])
    sentiment = sentiment[['date', 'value', 'classification']].rename(columns={'classification': 'sentiment'})

    trades['date'] = pd.to_datetime(trades['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True).dt.normalize()
    trades.columns = [c.strip().lower().replace(' ', '_') for c in trades.columns]
    trades.rename(columns={'execution_price': 'price', 'closed_pnl': 'closed_pnl',
                           'start_position': 'start_position'}, inplace=True)

    merged = trades.merge(sentiment, on='date', how='left').dropna(subset=['sentiment'])
    sentiment_map = {'Extreme Fear': 'Fear', 'Fear': 'Fear',
                     'Neutral': 'Greed', 'Greed': 'Greed', 'Extreme Greed': 'Greed'}
    merged['sentiment_binary'] = merged['sentiment'].map(sentiment_map)
    merged['is_win']      = merged['closed_pnl'] > 0
    merged['is_long']     = merged['direction'].str.lower() == 'buy'
    merged['leverage_proxy'] = merged.apply(
        lambda r: r['size_usd'] / abs(r['start_position']) if abs(r['start_position']) > 1 else np.nan, axis=1)

    daily = merged.groupby(['account', 'date', 'sentiment_binary']).agg(
        daily_pnl    = ('closed_pnl', 'sum'),
        trade_count  = ('closed_pnl', 'count'),
        win_count    = ('is_win', 'sum'),
        avg_size_usd = ('size_usd', 'mean'),
        avg_leverage = ('leverage_proxy', 'mean'),
        long_count   = ('is_long', 'sum'),
    ).reset_index()
    daily['win_rate']   = daily['win_count'] / daily['trade_count']
    daily['long_ratio'] = daily['long_count'] / daily['trade_count']

    trader_stats = merged.groupby('account').agg(
        total_pnl        = ('closed_pnl', 'sum'),
        total_trades     = ('closed_pnl', 'count'),
        win_rate         = ('is_win', 'mean'),
        avg_size_usd     = ('size_usd', 'mean'),
        avg_leverage     = ('leverage_proxy', 'mean'),
        long_ratio       = ('is_long', 'mean'),
        pnl_std          = ('closed_pnl', 'std'),
    ).reset_index()
    trader_stats['avg_daily_trades'] = trader_stats['total_trades'] / merged['date'].nunique()

    return merged, daily, trader_stats

with st.spinner("⚡ Loading data..."):
    merged, daily, trader_stats = load_data()

# sidebar
with st.sidebar:
    st.markdown("## 🎛️ Controls")
    st.markdown("---")
    sentiment_filter = st.multiselect(
        "📊 Sentiment Filter",
        ["Fear", "Greed"],
        default=["Fear", "Greed"]
    )
    date_range = st.date_input(
        "📅 Date Range",
        value=[merged['date'].min(), merged['date'].max()],
        min_value=merged['date'].min(),
        max_value=merged['date'].max()
    )
    st.markdown("---")
    st.markdown("### 📌 Dataset Overview")
    st.markdown(f"**Total Trades:** `{merged.shape[0]:,}`")
    st.markdown(f"**Unique Traders:** `{merged['account'].nunique()}`")
    st.markdown(f"**Unique Coins:** `{merged['coin'].nunique()}`")
    st.markdown(f"**Date Range:** `May 2023 — May 2025`")
    st.markdown("---")
    st.markdown("### 🏆 Key Numbers")
    st.markdown("**Consistent Winners:** `9 / 32`")
    st.markdown("**Model Accuracy:** `66.45%`")
    st.markdown("**Transition Alpha:** `+55% Mean PnL`")

filtered = daily[daily['sentiment_binary'].isin(sentiment_filter)]
if len(date_range) == 2:
    filtered = filtered[
        (filtered['date'] >= pd.Timestamp(date_range[0])) &
        (filtered['date'] <= pd.Timestamp(date_range[1]))
    ]

# tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Performance",
    "🧠 Behavior",
    "👥 Segments",
    "🔮 Model & Archetypes",
    "⚡ Regime Transitions"
])

# TAB 1 — Performance
with tab1:
    st.subheader("Performance: Fear vs Greed Days")

    fear_pnl  = filtered[filtered['sentiment_binary']=='Fear']['daily_pnl'].median()
    greed_pnl = filtered[filtered['sentiment_binary']=='Greed']['daily_pnl'].median()
    fear_wr   = filtered[filtered['sentiment_binary']=='Fear']['win_rate'].mean()
    greed_wr  = filtered[filtered['sentiment_binary']=='Greed']['win_rate'].mean()
    fear_tc   = filtered[filtered['sentiment_binary']=='Fear']['trade_count'].mean()
    greed_tc  = filtered[filtered['sentiment_binary']=='Greed']['trade_count'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("😨 Fear Median PnL",  f"${fear_pnl:,.0f}")
    col2.metric("😎 Greed Median PnL", f"${greed_pnl:,.0f}", f"+{greed_pnl-fear_pnl:.0f} vs Fear")
    col3.metric("😨 Fear Win Rate",    f"{fear_wr:.1%}")
    col4.metric("😎 Greed Win Rate",   f"{greed_wr:.1%}", f"+{(greed_wr-fear_wr):.1%} vs Fear")

    col1, col2 = st.columns(2)
    with col1:
        pnl_data = filtered.groupby('sentiment_binary')['daily_pnl'].median().reset_index()
        fig = px.bar(pnl_data, x='sentiment_binary', y='daily_pnl',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title="Median Daily PnL by Sentiment",
                     text='daily_pnl', labels={'daily_pnl': 'PnL (USD)', 'sentiment_binary': 'Sentiment'})
        fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        wr_data = filtered.groupby('sentiment_binary')['win_rate'].mean().reset_index()
        fig2 = px.bar(wr_data, x='sentiment_binary', y='win_rate',
                      color='sentiment_binary',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                      title="Average Win Rate by Sentiment",
                      text='win_rate', labels={'win_rate': 'Win Rate', 'sentiment_binary': 'Sentiment'})
        fig2.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig2.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font_color='white', yaxis_range=[0, 1])
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Greed days produce <b>2x higher median PnL</b> ($243 vs $123) — but win rate is nearly identical (~36%), meaning sentiment affects <i>how much</i> traders earn, not <i>how often</i> they win.</div>', unsafe_allow_html=True)

    # timeline
    st.markdown("#### 📅 PnL Over Time")
    timeline = merged.groupby(['date', 'sentiment_binary'])['closed_pnl'].mean().reset_index()
    timeline = timeline.groupby('date').first().reset_index()
    fig3 = px.line(timeline, x='date', y='closed_pnl',
                   color='sentiment_binary',
                   color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                   title="Average Daily PnL Over Time",
                   labels={'closed_pnl': 'Avg PnL (USD)', 'date': 'Date'})
    fig3.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
    st.plotly_chart(fig3, use_container_width=True)

# TAB 2 — Behavior
with tab2:
    st.subheader("Trader Behavior: Fear vs Greed Days")

    col1, col2, col3 = st.columns(3)
    fear_lev  = filtered[filtered['sentiment_binary']=='Fear']['avg_leverage'].median()
    greed_lev = filtered[filtered['sentiment_binary']=='Greed']['avg_leverage'].median()
    fear_lr   = filtered[filtered['sentiment_binary']=='Fear']['long_ratio'].mean()
    greed_lr  = filtered[filtered['sentiment_binary']=='Greed']['long_ratio'].mean()
    fear_sz   = filtered[filtered['sentiment_binary']=='Fear']['avg_size_usd'].median()
    greed_sz  = filtered[filtered['sentiment_binary']=='Greed']['avg_size_usd'].median()

    col1.metric("Fear Leverage",    f"{fear_lev:.2f}x", f"+{fear_lev-greed_lev:.2f}x vs Greed")
    col2.metric("Fear Long Ratio",  f"{fear_lr:.1%}",   f"{fear_lr-greed_lr:.1%} vs Greed")
    col3.metric("Fear Position Size", f"${fear_sz:,.0f}", f"${fear_sz-greed_sz:,.0f} vs Greed")

    col1, col2, col3 = st.columns(3)
    with col1:
        lev_data = filtered.groupby('sentiment_binary')['avg_leverage'].median().reset_index()
        fig = px.bar(lev_data, x='sentiment_binary', y='avg_leverage',
                     color='sentiment_binary',
                     color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                     title="Median Leverage by Sentiment", text='avg_leverage',
                     labels={'avg_leverage': 'Leverage', 'sentiment_binary': 'Sentiment'})
        fig.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        lr_data = filtered.groupby('sentiment_binary')['long_ratio'].mean().reset_index()
        fig2 = px.bar(lr_data, x='sentiment_binary', y='long_ratio',
                      color='sentiment_binary',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                      title="Average Long Ratio by Sentiment", text='long_ratio',
                      labels={'long_ratio': 'Long Ratio', 'sentiment_binary': 'Sentiment'})
        fig2.update_traces(texttemplate='%{text:.1%}', textposition='outside')
        fig2.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        sz_data = filtered.groupby('sentiment_binary')['avg_size_usd'].median().reset_index()
        fig3 = px.bar(sz_data, x='sentiment_binary', y='avg_size_usd',
                      color='sentiment_binary',
                      color_discrete_map={'Fear': '#e74c3c', 'Greed': '#2ecc71'},
                      title="Median Position Size by Sentiment", text='avg_size_usd',
                      labels={'avg_size_usd': 'Size (USD)', 'sentiment_binary': 'Sentiment'})
        fig3.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig3.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Traders use <b>higher leverage on Fear days</b> (4.39x vs 3.10x) while going more short (long ratio drops from 11.5% to 8.6%) — riskier behavior during the most volatile periods.</div>', unsafe_allow_html=True)

# TAB 3 — Segments
with tab3:
    st.subheader("Trader Segments & Heatmap")

    seg_data = pd.DataFrame({
        'Segment': ['High Leverage', 'Low Leverage', 'Frequent', 'Infrequent', 'Consistent Winner', 'Inconsistent'],
        'Fear PnL': [202, 107, 196, 0, 303, 54],
        'Greed PnL': [67, 349, 389, 0, 697, 120]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(seg_data, x='Segment', y=['Fear PnL', 'Greed PnL'],
                     barmode='group',
                     color_discrete_map={'Fear PnL': '#e74c3c', 'Greed PnL': '#2ecc71'},
                     title="Median Daily PnL by Segment & Sentiment")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                          font_color='white', xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        heatmap_data = seg_data.set_index('Segment')
        fig2 = px.imshow(heatmap_data,
                         color_continuous_scale='RdYlGn',
                         title="Heatmap: Segment × Sentiment PnL",
                         text_auto=True, aspect='auto')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> High leverage traders earn <b>3x more on Fear days</b> ($202 vs $67). Low leverage traders earn <b>3x more on Greed days</b> ($349 vs $107). Consistent winners profit on BOTH.</div>', unsafe_allow_html=True)

    st.markdown("#### 🔍 Explore Individual Traders")
    selected_trader = st.selectbox("Select Trader", trader_stats['account'].tolist())
    t = trader_stats[trader_stats['account'] == selected_trader].iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total PnL",      f"${t['total_pnl']:,.0f}")
    col2.metric("Win Rate",       f"{t['win_rate']:.1%}")
    col3.metric("Avg Leverage",   f"{t['avg_leverage']:.1f}x")
    col4.metric("Avg Daily Trades", f"{t['avg_daily_trades']:.1f}")

# TAB 4 — Model
with tab4:
    st.subheader("Predictive Model & Trader Archetypes")

    col1, col2, col3 = st.columns(3)
    col1.metric("🎯 Model Accuracy",    "66.45%")
    col2.metric("✅ Profit Precision",  "69%")
    col3.metric("📡 Profit Recall",     "84%")

    col1, col2 = st.columns(2)
    with col1:
        importance = pd.DataFrame({
            'Feature':    ['Win Rate', 'Avg Leverage', 'Avg Size USD', 'Trade Count', 'Long Ratio', 'Sentiment'],
            'Importance': [0.298, 0.270, 0.186, 0.163, 0.068, 0.015]
        }).sort_values('Importance')
        fig = px.bar(importance, x='Importance', y='Feature', orientation='h',
                     color='Importance', color_continuous_scale='Blues',
                     title="Feature Importance — GBM Model")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        archetypes = pd.DataFrame({
            'Archetype':      ['Hyperactive', 'Consistent Performer', 'Balanced', 'Underperformer'],
            'Win Rate':       [0.40, 0.48, 0.38, 0.42],
            'Avg Leverage':   [499, 3094, 224, 149],
            'Trades/Day':     [38.5, 11.9, 10.1, 4.9],
            'Total PnL':      [1272055, 183079, 158049, 79175]
        })
        fig2 = px.scatter(archetypes, x='Avg Leverage', y='Win Rate',
                          size='Trades/Day', color='Archetype', text='Archetype',
                          title="Trader Archetypes: Leverage vs Win Rate",
                          log_x=True, size_max=60)
        fig2.update_traces(textposition='top center')
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="insight-box">💡 <b>Key Insight:</b> Win rate (0.298) and leverage (0.270) are the strongest predictors of next-day profitability. Sentiment alone is a weak signal (0.015) — it works <i>indirectly</i> through behavior.</div>', unsafe_allow_html=True)

# TAB 5 — Regime Transitions
with tab5:
    st.subheader("⚡ Regime Transition Analysis — The Hidden Alpha")
    st.markdown("> *Most analyses compare Fear vs Greed in isolation. The real alpha lies at the **moment sentiment switches.***")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Normal Day Mean PnL",     "$4,112")
    col2.metric("Transition Day Mean PnL", "$6,363", "+55% 🚀")
    col3.metric("Fear→Greed Mean PnL",     "$6,581", "Most Profitable")
    col4.metric("Greed→Fear Mean PnL",     "$6,177", "Still High")

    transition_data = pd.DataFrame({
        'Type':       ['Normal Day', 'Transition Day', 'Fear→Greed', 'Greed→Fear'],
        'Median PnL': [209, 196, 205, 170],
        'Mean PnL':   [4112, 6363, 6581, 6177],
        'Days':       [2059, 281, 145, 135]
    })

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(transition_data[transition_data['Type'].isin(['Normal Day', 'Transition Day'])],
                     x='Type', y='Mean PnL',
                     color='Type',
                     color_discrete_map={'Normal Day': '#3498db', 'Transition Day': '#e74c3c'},
                     title="Mean PnL: Normal vs Transition Days", text='Mean PnL')
        fig.update_traces(texttemplate='$%{text:,}', textposition='outside')
        fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(transition_data[transition_data['Type'].isin(['Fear→Greed', 'Greed→Fear'])],
                      x='Type', y='Mean PnL',
                      color='Type',
                      color_discrete_map={'Fear→Greed': '#2ecc71', 'Greed→Fear': '#e74c3c'},
                      title="Mean PnL by Transition Type", text='Mean PnL')
        fig2.update_traces(texttemplate='$%{text:,}', textposition='outside')
        fig2.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)',
                           paper_bgcolor='rgba(0,0,0,0)', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="strategy-box">🎯 <b>Strategy 3 — Regime Transition Trading</b><br>• Monitor sentiment daily for Fear↔Greed switches<br>• On transition day + 2 days after: increase position sizing for high-leverage, frequent traders<br>• Fear→Greed transitions produce the highest mean PnL ($6,581) in the entire dataset<br>• This is how <b>institutional quant funds</b> think — not static regimes, but <b>regime transitions</b></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("*Built by **Rushikesh Baban Kedar** | Primetrade.ai Data Science Internship | 📧 rushikedar40@gmail.com*")