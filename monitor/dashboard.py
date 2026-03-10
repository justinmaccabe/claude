# monitor/dashboard.py  v2  —  run: streamlit run monitor/dashboard.py
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats as scipy_stats
import streamlit as st
from datetime import datetime, timezone

from config import TICKERS, DATA, MARKET_VOL_TARGET, CASH_RATE_REAL
from data.store import Store
from data.fetcher import fetch_news
from signals.trend import TrendSignal
from signals.carry import CarrySignal
from signals.regime import RegimeSignal, VIX_CALM, VIX_NEUTRAL, VIX_STRESS
from portfolio.allocator import Allocator, SignalOverlay, generate_trades
from jobs.weekly import EXPECTED_RETURNS, VOLS, CORR_MATRIX, TICKERS_LIST
from monitor.holdings import load as load_holdings, initialize_to_target, update_weights, get_trade_log

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARKS  = {"SPY": "#f5d76e", "VTI": "#a855f7"}
SLEEVE_COL  = {"core": "#4A9EDD", "factor": "#F5D76E"}
TICKER_COL  = {t: SLEEVE_COL[TICKERS[t]["sleeve"]] for t in TICKERS}
REGIME_COL  = {"CALM":"#10b981","NEUTRAL":"#4a9edd","ELEVATED":"#f5a623","STRESS":"#e05c3a"}
REGIME_ICO  = {"CALM":"🟢","NEUTRAL":"🔵","ELEVATED":"🟠","STRESS":"🔴"}
REGIME_DESC = {
    "CALM":     f"VIX <{VIX_CALM} — markets quiet. Tilting toward momentum, EM and high yield.",
    "NEUTRAL":  f"VIX {VIX_CALM}-{VIX_NEUTRAL} — normal vol. Running base weights, no tilt.",
    "ELEVATED": f"VIX {VIX_NEUTRAL}-{VIX_STRESS} — vol rising. Mild tilt toward quality and duration.",
    "STRESS":   f"VIX >{VIX_STRESS} — market stress. Full tilt to quality, TIPS and treasuries.",
}
GRID = dict(gridcolor="#162030", zerolinecolor="#162030")

# Known ETF expense ratios (bps)
ETF_ER = {"SPY":0.0945,"EFA":0.32,"EEM":0.68,"IEF":0.15,"SCHP":0.03,
           "HYG":0.48,"QMOM":0.49,"QUAL":0.15,"AVUV":0.25,"KMLM":0.90,"DBMF":0.85}
# Known ETF dividend yields (approx %)
ETF_YIELD = {"SPY":1.3,"EFA":3.1,"EEM":2.5,"IEF":3.8,"SCHP":3.2,
             "HYG":6.5,"QMOM":0.4,"QUAL":1.1,"AVUV":1.5,"KMLM":0.0,"DBMF":0.0}

# Historical crisis episodes  {name: (start, end, label)}
CRISES = {
    "GFC 2008-09":        ("2008-09-01","2009-03-31","S&P -55%"),
    "COVID Crash 2020":   ("2020-02-19","2020-03-23","S&P -34%"),
    "Rate Shock 2022":    ("2022-01-03","2022-10-12","S&P -25%"),
    "Dot-com 2000-02":    ("2000-03-24","2002-10-09","S&P -49%"),
    "Taper Tantrum 2013": ("2013-05-22","2013-06-24","S&P -6%"),
    "Q4 Selloff 2018":    ("2018-10-03","2018-12-24","S&P -20%"),
}

def PL(**kw):
    base = dict(paper_bgcolor="#080c14", plot_bgcolor="#0d1525",
                font=dict(family="monospace", color="#8da0bc", size=11),
                margin=dict(l=10, r=10, t=40, b=10),
                legend=dict(bgcolor="#0d1525", bordercolor="#162030", borderwidth=1))
    base.update(kw)
    return base

st.set_page_config(page_title="Portfolio System v2", page_icon="📊", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
.block-container{padding:1.2rem 2rem 2rem}
.stMetric{background:#0d1525;border:1px solid #162030;border-radius:6px;padding:10px 14px}
.stMetric label{font-size:9px!important;letter-spacing:1px}
</style>""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_store():
    return Store(DATA["db_path"])

@st.cache_data(ttl=3600)
def load_all():
    store  = get_store()
    prices = store.read_prices(tickers=TICKERS_LIST + list(BENCHMARKS.keys()))
    trend  = TrendSignal(prices)
    carry  = CarrySignal(prices, store=store)
    regime = RegimeSignal(store=store, prices=prices)

    trend_df   = trend.signal()
    carry_df   = carry.signal()
    regime_det = regime.detect()
    vix        = regime.current_vix()
    vix_pct    = regime.vix_percentile()
    vix_hist   = regime.vix_history(days=365)
    tilts      = regime.tilts()

    macro = {}
    for key in ["hyg_oas","breakeven","real_yield","t2y","t10y","t30y","usdcad","ca10y"]:
        try:
            s = store.read_macro(key)
            if s is not None and not s.empty:
                macro[key] = s
        except:
            pass

    trend_signals = pd.Series(dtype=float)
    if not trend_df.empty:
        trend_signals = trend_df["signal"].reindex(TICKERS_LIST).fillna(0)

    carry_signals = pd.Series(dtype=float)
    if not carry_df.empty and "HYG  (credit spread)" in carry_df.index:
        carry_signals = pd.Series({"HYG": carry_df.loc["HYG  (credit spread)","signal"]})

    overlay   = SignalOverlay(trend_signals=trend_signals,carry_signals=carry_signals,regime_tilts=tilts)
    allocator = Allocator(EXPECTED_RETURNS, VOLS, CORR_MATRIX)
    base_w    = allocator.base_weights()
    target_w  = allocator.target_weights(overlay=overlay, levered=True)
    stats     = allocator.portfolio_stats(target_w)
    base_stats= allocator.portfolio_stats(base_w)

    return dict(prices=prices, trend_df=trend_df, carry_df=carry_df, macro=macro,
                regime=regime_det, vix=vix, vix_pct=vix_pct, vix_hist=vix_hist, tilts=tilts,
                base_w=base_w, target_w=target_w, stats=stats, base_stats=base_stats)

@st.cache_data(ttl=1800)
def load_news():
    return fetch_news(TICKERS_LIST)

with st.spinner("Loading..."):
    d = load_all()

if d["prices"].empty:
    st.error("No data. Run: python3 -m data.fetcher"); st.stop()

prices   = d["prices"]
target_w = d["target_w"]

# ── Core helpers ──────────────────────────────────────────────────────────────
def port_returns(w):
    av = [t for t in TICKERS_LIST if t in prices.columns]
    wn = w.reindex(av).fillna(0)
    wn = wn / wn.sum() if wn.sum() > 0 else wn
    return prices[av].pct_change(fill_method=None).dropna().dot(wn)

def cum(r): return (1+r).cumprod()

def ann_stats(r):
    if len(r) < 2: return dict(ret=0,vol=0,sr=0,dd=0,calmar=0,sortino=0)
    ret = r.mean()*252*100; vol = r.std()*np.sqrt(252)*100
    sr  = (ret-CASH_RATE_REAL)/vol if vol>0 else 0
    cr  = cum(r); dd = (cr/cr.cummax()-1).min()*100
    calmar = -ret/dd if dd<0 else 0
    neg    = r[r<0]; sortino = (ret-CASH_RATE_REAL)/(neg.std()*np.sqrt(252)*100) if len(neg)>0 else 0
    return dict(ret=ret, vol=vol, sr=sr, dd=dd, calmar=calmar, sortino=sortino)

def period_ret(r, n):
    sl = r.iloc[-n:] if len(r)>=n else r
    return ((1+sl).prod()-1)*100

port_ret = port_returns(target_w)
dp = dict(
    day  = period_ret(port_ret, 1),
    week = period_ret(port_ret, 5),
    mtd  = period_ret(port_ret[port_ret.index.month==port_ret.index[-1].month], len(port_ret)),
    ytd  = period_ret(port_ret[port_ret.index.year==port_ret.index[-1].year], len(port_ret)),
)
ps_all = ann_stats(port_ret)

# ── Header ────────────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([6,1])
with hc1:
    st.markdown("<span style='font-size:10px;color:#2d5a8a;letter-spacing:3px'>AQR 2026 CMA · PORTFOLIO SYSTEM v2</span><br>"
                "<span style='font-size:26px;font-weight:700;color:#ddeeff'>Signal Dashboard</span>", unsafe_allow_html=True)
with hc2:
    if st.button("↻ Refresh", width='stretch'):
        st.cache_data.clear(); st.rerun()

st.caption(f"Updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ·  "
           f"{len(TICKERS_LIST)} positions  ·  Target vol {MARKET_VOL_TARGET}%")

s, b   = d["stats"], d["base_stats"]
rv     = d["regime"].value
rc     = REGIME_COL.get(rv, "#4a9edd")
ri     = REGIME_ICO.get(rv, "🔵")

spy_ytd_delta = "—"
if "SPY" in prices.columns:
    spy_r   = prices["SPY"].pct_change(fill_method=None).dropna()
    spy_ytd = period_ret(spy_r[spy_r.index.year==spy_r.index[-1].year], len(spy_r))
    spy_ytd_delta = f"{dp['ytd']-spy_ytd:+.2f}%"

# Portfolio cost & yield
wt_er    = sum(ETF_ER.get(t,0)*float(target_w.get(t,0)) for t in TICKERS_LIST)
wt_yield = sum(ETF_YIELD.get(t,0)*float(target_w.get(t,0)) for t in TICKERS_LIST)

# 12 metrics — short labels to avoid truncation
m = st.columns(12)
m[0].metric("SHARPE",      f"{s['sharpe_ratio']:.3f}",    f"{s['sharpe_ratio']-b['sharpe_ratio']:+.3f} base")
m[1].metric("EXP RET",     f"{s['expected_return']:.1f}%", f"base {b['expected_return']:.1f}%")
m[2].metric("HIST RET",    f"{ps_all['ret']:.1f}%",         "3yr ann.")
m[3].metric("VOL",         f"{ps_all['vol']:.1f}%",         f"tgt {MARKET_VOL_TARGET}%")
m[4].metric("LEVERAGE",    f"{s['gross_exposure']:.2f}×")
m[5].metric("MAX DD",      f"{ps_all['dd']:.1f}%",          f"sortino {ps_all['sortino']:.2f}")
m[6].metric(f"{ri} REGIME",f"{rv}",                         f"VIX {d['vix']:.1f}" if d["vix"] else None)
m[7].metric("TODAY",       f"{dp['day']:+.2f}%",             delta_color="normal")
m[8].metric("WEEK",        f"{dp['week']:+.2f}%",            delta_color="normal")
m[9].metric("MTD",         f"{dp['mtd']:+.2f}%",             delta_color="normal")
m[10].metric("YTD",        f"{dp['ytd']:+.2f}%",             delta_color="normal")
m[11].metric("VS SPY YTD", spy_ytd_delta,                    delta_color="normal")

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

tabs = st.tabs([
    "⚖️ ALLOCATION","📡 SIGNALS","📈 PERFORMANCE",
    "📊 VS BENCHMARK","🔬 ANALYZER",
    "💥 STRESS TEST","🎲 MONTE CARLO",
    "🧬 FACTOR REGRESSION","🔁 REBALANCE","🗺️ FRONTIER"
])
tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10 = tabs

# ══ TAB 1: ALLOCATION ════════════════════════════════════════════════════════
with tab1:
    positions = sorted([
        {"ticker":t,"label":TICKERS[t]["label"],"sleeve":TICKERS[t]["sleeve"],
         "weight":float(target_w.get(t,0)),"exp_ret":EXPECTED_RETURNS[t],
         "vol":VOLS[t],"sharpe":round((EXPECTED_RETURNS[t]-CASH_RATE_REAL)/VOLS[t],2),
         "er":ETF_ER.get(t,0),"div":ETF_YIELD.get(t,0),"colour":TICKER_COL[t]}
        for t in TICKERS_LIST], key=lambda x: x["weight"], reverse=True)

    # Cost & yield summary
    ca,cb,cc,cd = st.columns(4)
    ca.metric("Wtd Exp Ratio", f"{wt_er:.2f}%", "annual drag")
    cb.metric("Wtd Div Yield", f"{wt_yield:.2f}%", "blended")
    cc.metric("Est. Annual Income", f"${wt_yield/100*100000:,.0f}", "on $100k")
    cd.metric("Net Expected Ret",   f"{s['expected_return']-wt_er:.1f}%", "after fees")

    c1,c2 = st.columns([3,2])
    with c1:
        fig=go.Figure()
        for sl in ["core","factor"]:
            pts=[p for p in positions if p["sleeve"]==sl]
            fig.add_trace(go.Bar(y=[p["ticker"] for p in pts],x=[p["weight"]*100 for p in pts],
                name=sl.upper(),orientation="h",marker_color=SLEEVE_COL[sl],marker_line_width=0,
                text=[f"{p['weight']*100:.1f}%" for p in pts],textposition="outside",
                textfont=dict(color="#8da0bc",size=10)))
        fig.update_layout(**PL(title=dict(text="🎯 Target Weights",font=dict(color="#ddeeff",size=12)),
                               barmode="group",height=360))
        fig.update_xaxes(title="Weight (%)",ticksuffix="%",**GRID)
        fig.update_yaxes(categoryorder="total ascending",**GRID)
        st.plotly_chart(fig,width='stretch')
    with c2:
        st_totals={sl.upper():sum(p["weight"]*100 for p in positions if p["sleeve"]==sl)
                   for sl in ["core","factor"]}
        fig2=go.Figure(go.Pie(labels=list(st_totals.keys()),values=list(st_totals.values()),
            hole=0.6,marker_colors=[SLEEVE_COL["core"],SLEEVE_COL["factor"]],
            textinfo="label+percent",textfont=dict(color="#ddeeff",size=12)))
        fig2.add_annotation(text=f"<b>{s['gross_exposure']:.2f}×</b><br>leverage",
                            x=0.5,y=0.5,showarrow=False,font=dict(size=13,color="#ddeeff"))
        fig2.update_layout(**PL(title=dict(text="Sleeve",font=dict(color="#ddeeff",size=12)),height=240))
        st.plotly_chart(fig2,width='stretch')
        fig3=go.Figure()
        for p in positions:
            fig3.add_trace(go.Scatter(x=[p["vol"]],y=[p["exp_ret"]],mode="markers+text",
                marker=dict(size=max(p["weight"]*400,8),color=p["colour"],opacity=0.85,
                            line=dict(color="#080c14",width=1)),
                text=[p["ticker"]],textposition="top center",textfont=dict(color="#ddeeff",size=9),
                showlegend=False,
                hovertemplate=f"<b>{p['ticker']}</b><br>Vol:{p['vol']}%  Ret:{p['exp_ret']}%  "
                              f"Wt:{p['weight']*100:.1f}%  ER:{p['er']}%  Yield:{p['div']}%<extra></extra>"))
        fig3.update_layout(**PL(title=dict(text="Risk vs Return",font=dict(color="#ddeeff",size=12)),height=240))
        fig3.update_xaxes(title="Vol (%)",**GRID); fig3.update_yaxes(title="Exp Ret (%)",**GRID)
        st.plotly_chart(fig3,width='stretch')

    df_pos=pd.DataFrame(positions)
    for col,fmt in [("weight","{:.1%}"),("exp_ret","{:.1f}%"),("vol","{:.1f}%"),
                    ("sharpe","{:.2f}"),("er","{:.2f}%"),("div","{:.1f}%")]:
        df_pos[col]=df_pos[col].map(fmt.format)
    st.dataframe(df_pos[["ticker","label","sleeve","exp_ret","vol","sharpe","weight","er","div"]].rename(
        columns={"ticker":"Ticker","label":"Name","sleeve":"Sleeve","exp_ret":"Exp Ret","vol":"Vol",
                 "sharpe":"Sharpe","weight":"Weight","er":"Exp Ratio","div":"Div Yield"}
    ).set_index("Ticker"),width='stretch',height=280)

# ══ TAB 2: SIGNALS ═══════════════════════════════════════════════════════════
with tab2:
    macro=d["macro"]
    prices_all=d["prices"]
    tdf=d["trend_df"]
    tilts=d["tilts"]

    # ── Top metrics bar ───────────────────────────────────────────────────────
    sc=st.columns(6)
    if not tdf.empty:
        n_bull=int((tdf["signal"]>0).sum()); avg_z=float(tdf["z_score"].mean())
        strongest=tdf["z_score"].idxmax(); weakest=tdf["z_score"].idxmin()
        sc[0].metric("🐂 Bulls",    f"{n_bull}/{len(tdf)}",delta_color="off")
        sc[1].metric("📊 Avg Z",    f"{avg_z:+.2f}σ",delta_color="normal")
        sc[2].metric("🏆 Strongest",strongest,f"{tdf.loc[strongest,'z_score']:+.2f}σ",delta_color="off")
        sc[3].metric("🐌 Weakest",  weakest,  f"{tdf.loc[weakest,'z_score']:+.2f}σ",delta_color="off")
    else:
        sc[0].metric("🐂 Bulls","—"); sc[1].metric("📊 Avg Z","—")
        sc[2].metric("🏆 Strongest","—"); sc[3].metric("🐌 Weakest","—")
    sc[4].metric(f"{ri} Regime", rv, f"VIX {d['vix']:.1f}" if d["vix"] else None)
    sc[5].metric("📍 VIX %ile", f"{d['vix_pct']:.0f}th" if d["vix_pct"] else "—",delta_color="off")

    # ── Plain-English signal summary card ─────────────────────────────────────
    if not tdf.empty:
        bulls=[t for t in tdf.index if tdf.loc[t,"signal"]>0]
        bears=[t for t in tdf.index if tdf.loc[t,"signal"]<0]
        strong_bulls=[t for t in tdf.index if tdf.loc[t,"z_score"]>1]
        strong_bears=[t for t in tdf.index if tdf.loc[t,"z_score"]<-1]
        bull_str  = ", ".join(bulls)  if bulls  else "none"
        bear_str  = ", ".join(bears)  if bears  else "none"
        sb_str    = ", ".join(strong_bulls) if strong_bulls else "none"
        sw_str    = ", ".join(strong_bears) if strong_bears else "none"

        tilt_ups   = [f"{t} (+{v:.0%})" for t,v in tilts.items() if v>0]
        tilt_downs = [f"{t} ({v:.0%})"  for t,v in tilts.items() if v<0]
        tilt_line  = ""
        if tilt_ups or tilt_downs:
            tilt_line = (f"The {rv} regime is nudging weights <b style='color:#10b981'>up</b> on "
                        f"{', '.join(tilt_ups) or '—'} and <b style='color:#e05c3a'>down</b> on "
                        f"{', '.join(tilt_downs) or '—'}.")

        if n_bull >= len(tdf)*0.7:
            mood="🟢 Broadly positive"; mood_sub="Most of your holdings have upward momentum. The portfolio is in a good place trend-wise."
            mood_col="#10b981"
        elif n_bull >= len(tdf)*0.4:
            mood="🟡 Mixed signals"; mood_sub="Some positions trending well, others fading. Selectivity matters right now."
            mood_col="#f5a623"
        else:
            mood="🔴 Mostly weak"; mood_sub="Momentum has faded broadly. The system is leaning defensive."
            mood_col="#e05c3a"

        vix_line=""
        if d["vix"]:
            if d["vix"]>VIX_STRESS:
                vix_line=f"⚠️ <b>VIX is {d['vix']:.1f}</b> — markets are scared right now. Your system has automatically shifted into defensive mode, boosting quality and duration and trimming risk assets."
            elif d["vix"]>VIX_NEUTRAL:
                vix_line=f"🟠 <b>VIX at {d['vix']:.1f}</b> — elevated but not panicking. The system has a mild defensive lean."
            else:
                vix_line=f"🟢 <b>VIX at {d['vix']:.1f}</b> — markets are calm. Risk-on tilt is active."

        # Bottom-line action sentence
        if strong_bulls and rv in ("STRESS","ELEVATED"):
            action=f"📌 <b>Bottom line:</b> Despite strong momentum in {', '.join(strong_bulls)}, the stressed VIX regime is holding back risk exposure. Watch for VIX to drop below {VIX_NEUTRAL} before leaning in harder."
        elif strong_bulls and rv=="CALM":
            action=f"📌 <b>Bottom line:</b> Strong momentum + calm market = the system is leaning into {', '.join(strong_bulls)}. Conditions are aligned."
        elif strong_bears:
            action=f"📌 <b>Bottom line:</b> {', '.join(strong_bears)} are showing unusually weak momentum — the system is underweighting these. Worth monitoring."
        else:
            action="📌 <b>Bottom line:</b> No extreme signals either way. The system is holding near its base allocation."

        st.markdown(f"""<div style='background:linear-gradient(135deg,#0d1525,#0a1020);
            border:1px solid #1e3050;border-left:4px solid {mood_col};border-radius:10px;
            padding:18px 22px;margin:10px 0 16px 0'>
            <div style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-bottom:10px'>
            🧠 WHAT THE SIGNALS ARE TELLING YOU RIGHT NOW</div>
            <div style='font-size:15px;font-weight:700;color:{mood_col};margin-bottom:4px'>{mood}</div>
            <div style='font-size:12px;color:#ddeeff;line-height:1.7;margin-bottom:10px'>{mood_sub}</div>
            <div style='font-size:12px;color:#8da0bc;line-height:1.9'>
            📈 <b style='color:#10b981'>Trending up:</b> {bull_str}<br>
            📉 <b style='color:#e05c3a'>Trending down:</b> {bear_str}<br>
            💪 <b style='color:#f5d76e'>High conviction (above +1σ):</b> {sb_str}<br>
            😬 <b style='color:#f5a623'>Weak momentum (below −1σ):</b> {sw_str}
            </div>
            <div style='font-size:11px;color:#8da0bc;line-height:1.7;margin-top:10px;padding-top:8px;border-top:1px solid #162030'>
            {vix_line}
            </div>
            <div style='font-size:11px;color:#ddeeff;line-height:1.7;margin-top:10px;padding-top:8px;border-top:1px solid #162030'>
            {action}
            </div>
            {f'<div style="font-size:10px;color:#4a6fa5;margin-top:10px;padding-top:8px;border-top:1px solid #0e1a2c">⚙️ {tilt_line}</div>' if tilt_line else ""}
        </div>""",unsafe_allow_html=True)

    # ── Regime banner ─────────────────────────────────────────────────────────
    st.markdown(f"""<div style='background:#0d1525;border-left:4px solid {rc};border-radius:6px;
        padding:10px 16px;margin:0 0 14px 0'>
        <span style='font-size:13px;font-weight:700;color:{rc}'>{ri} {rv} REGIME</span>
        <span style='font-size:11px;color:#8da0bc;margin-left:12px'>{REGIME_DESC.get(rv,"")}</span>
        <div style='font-size:9px;color:#4a6fa5;margin-top:4px'>
        🟢 &lt;{VIX_CALM} = CALM (risk-on)  ·  🔵 {VIX_CALM}–{VIX_NEUTRAL} = NEUTRAL  ·  🟠 {VIX_NEUTRAL}–{VIX_STRESS} = ELEVATED (mild defensive)  ·  🔴 &gt;{VIX_STRESS} = STRESS (full defensive)</div>
    </div>""",unsafe_allow_html=True)

    c1,c2=st.columns(2)
    with c1:
        # ── Momentum chart ────────────────────────────────────────────────────
        if not tdf.empty:
            with st.expander("📖 What is 12-1 Momentum and Z-Score?", expanded=False):
                st.markdown("""
**12-1 Momentum** looks at each ETF's return over the *past 12 months*, deliberately **skipping the most recent month**.
Why skip it? Short-term returns tend to reverse (mean reversion), so including last month actually *hurts* the signal's predictive power.
Academic research since 1993 (Jegadeesh & Titman) shows 12-1 is the sweet spot.

🟢 **Green bar** = positive trend = system wants to overweight this position
🔴 **Red bar** = negative trend = system wants to underweight or avoid

**Z-Score** tells you *how strong* the signal is versus its own history — like a confidence meter.
- Between the yellow lines (±1σ) = normal, nothing unusual
- **Beyond ±1σ** = unusually strong conviction — these are the ones to pay attention to
- Example: EEM at +1.52σ means its momentum is stronger than ~93% of all historical readings
                """)

            fig=go.Figure(go.Bar(x=tdf["momentum_12_1"]*100,y=tdf.index,orientation="h",
                marker_color=["#10b981" if s>0 else "#e05c3a" for s in tdf["signal"]],
                marker_line_width=0,text=[f"{v*100:+.1f}%" for v in tdf["momentum_12_1"]],
                textposition="outside",textfont=dict(color="#8da0bc",size=10)))
            fig.add_vline(x=0,line_color="#4a6fa5",line_dash="dot")
            fig.update_layout(**PL(height=300,title=dict(text="🚀 12-1 Momentum  (skip 1mo)",
                font=dict(color="#ddeeff",size=12))))
            fig.update_xaxes(title="12-month return, skip last month (%)",ticksuffix="%",**GRID)
            fig.update_yaxes(categoryorder="total ascending",**GRID)
            st.plotly_chart(fig,width='stretch')

            fig2=go.Figure(go.Bar(x=tdf["z_score"],y=tdf.index,orientation="h",
                marker_color=["#10b981" if z>0 else "#e05c3a" for z in tdf["z_score"]],
                marker_line_width=0,text=[f"{z:+.2f}σ" for z in tdf["z_score"]],
                textposition="outside",textfont=dict(color="#8da0bc",size=10)))
            fig2.add_vline(x=0,line_color="#4a6fa5",line_dash="dot")
            fig2.add_vline(x=1, line_color="#f5d76e",line_width=1,line_dash="dash",
                annotation_text="high conviction →",annotation_font_color="#f5d76e",annotation_font_size=8)
            fig2.add_vline(x=-1,line_color="#f5d76e",line_width=1,line_dash="dash",
                annotation_text="← high conviction",annotation_font_color="#f5d76e",annotation_font_size=8)
            fig2.update_layout(**PL(height=300,title=dict(text="📐 Signal Strength  (how unusual is this momentum?)",
                font=dict(color="#ddeeff",size=12))))
            fig2.update_xaxes(title="Standard deviations from historical average",**GRID)
            fig2.update_yaxes(categoryorder="total ascending",**GRID)
            st.plotly_chart(fig2,width='stretch')

            # ── Active tilts visual ───────────────────────────────────────────
            if tilts:
                st.markdown("""<div style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:6px'>
                    ⚙️ ACTIVE REGIME TILTS — HOW YOUR WEIGHTS ARE BEING ADJUSTED</div>""",
                    unsafe_allow_html=True)

                with st.expander("📖 What are regime tilts and why do they matter?", expanded=False):
                    st.markdown(f"""
The optimizer calculates your **base allocation** — the theoretically optimal weights given expected returns and correlations.

But the real world isn't static. When markets get scared (VIX spikes), holding the same weights as during calm times is naive.

**Regime tilts are automatic adjustments** on top of your base weights, triggered by the current VIX level:

| Regime | VIX | What happens |
|--------|-----|-------------|
| 🟢 CALM | <{VIX_CALM} | Lean into momentum & risk assets (EEM, HYG, QMOM) |
| 🔵 NEUTRAL | {VIX_CALM}–{VIX_NEUTRAL} | No adjustment — hold base weights |
| 🟠 ELEVATED | {VIX_NEUTRAL}–{VIX_STRESS} | Mild defensive shift |
| 🔴 STRESS | >{VIX_STRESS} | Full defensive: boost QUAL, SCHP, IEF — cut QMOM, EEM, HYG |

**Right now the {rv} regime is active**, so the adjustments below are live.
🟢 Green = your position is being *increased* vs the base optimizer output.
🔴 Red = your position is being *decreased* to reduce risk.
                    """)

                tilt_items=sorted(tilts.items(),key=lambda x:x[1],reverse=True)
                tilt_tickers=[t for t,_ in tilt_items]
                tilt_vals=[v for _,v in tilt_items]
                fig_t=go.Figure(go.Bar(
                    x=[v*100 for v in tilt_vals], y=tilt_tickers, orientation="h",
                    marker_color=["#10b981" if v>0 else "#e05c3a" for v in tilt_vals],
                    marker_line_width=0,
                    text=[f"{v:+.0%}" for v in tilt_vals],
                    textposition="outside",textfont=dict(color="#8da0bc",size=10),
                    hovertemplate="<b>%{y}</b>: %{x:+.1f}pp adjustment<extra></extra>"
                ))
                fig_t.add_vline(x=0,line_color="#4a6fa5",line_dash="dot")
                fig_t.update_layout(**PL(height=max(200,len(tilts)*40+60),
                    title=dict(text=f"⚙️ {ri} {rv} Regime — Weight Adjustments vs Base",
                    font=dict(color=rc,size=12))))
                fig_t.update_xaxes(title="Percentage-point adjustment to weight",ticksuffix="%",**GRID)
                fig_t.update_yaxes(**GRID)
                st.plotly_chart(fig_t,width='stretch')

            # ── Correlation changes ───────────────────────────────────────────
            st.markdown("""<div style='background:#0a1020;border:1px solid #162030;border-radius:6px;
                padding:10px 14px;margin:6px 0;font-size:11px;color:#8da0bc;line-height:1.6'>
                🔀 <b style='color:#ddeeff'>Are your diversifiers still working?</b><br>
                Shows how correlated pairs of your holdings have become over <b>30 days vs 90 days</b>.
                🔴 <b style='color:#e05c3a'>Red = converging</b> (moving more together = less diversification).
                🟢 <b style='color:#10b981'>Green = diverging</b> (moving more independently = better diversification).
                Big moves here can signal a regime shift before the VIX picks it up.
            </div>""",unsafe_allow_html=True)
            av_sig=[t for t in TICKERS_LIST if t in prices_all.columns]
            px_all=prices_all[av_sig].pct_change(fill_method=None).dropna()
            corr_30=px_all.iloc[-30:].corr(); corr_90=px_all.iloc[-90:].corr()
            corr_chg=corr_30-corr_90
            pairs=[]
            for i in range(len(av_sig)):
                for j in range(i+1,len(av_sig)):
                    a,bt=av_sig[i],av_sig[j]
                    pairs.append({"pair":f"{a}/{bt}","30d":round(corr_30.loc[a,bt],2),
                                  "90d":round(corr_90.loc[a,bt],2),"change":round(corr_chg.loc[a,bt],2)})
            pairs_df=pd.DataFrame(pairs).sort_values("change",key=abs,ascending=False).head(10)
            fig_cc=go.Figure(go.Bar(x=pairs_df["change"],y=pairs_df["pair"],orientation="h",
                marker_color=["#e05c3a" if v>0 else "#10b981" for v in pairs_df["change"]],
                marker_line_width=0,text=[f"{v:+.2f}" for v in pairs_df["change"]],
                textposition="outside",textfont=dict(color="#8da0bc",size=9),
                customdata=pairs_df[["30d","90d"]].values,
                hovertemplate="<b>%{y}</b><br>30d: %{customdata[0]:.2f}  ·  90d: %{customdata[1]:.2f}  ·  Δ: %{x:+.2f}<extra></extra>"))
            fig_cc.add_vline(x=0,line_color="#4a6fa5",line_dash="dot")
            fig_cc.update_layout(**PL(height=320,title=dict(
                text="🔀 Correlation Drift  (30d minus 90d)",font=dict(color="#ddeeff",size=12))))
            fig_cc.update_xaxes(title="Change in correlation (red = more correlated = worse diversification)",**GRID)
            fig_cc.update_yaxes(**GRID)
            st.plotly_chart(fig_cc,width='stretch')

    with c2:
        # ── VIX chart ─────────────────────────────────────────────────────────
        if d["vix_hist"] is not None:
            vh=d["vix_hist"]
            fig_v=go.Figure()
            fig_v.add_hrect(y0=0,          y1=VIX_CALM,   fillcolor="rgba(16,185,129,0.07)",line_width=0)
            fig_v.add_hrect(y0=VIX_CALM,   y1=VIX_NEUTRAL,fillcolor="rgba(74,158,221,0.07)",line_width=0)
            fig_v.add_hrect(y0=VIX_NEUTRAL,y1=VIX_STRESS,  fillcolor="rgba(245,166,35,0.07)",line_width=0)
            fig_v.add_hrect(y0=VIX_STRESS, y1=80,          fillcolor="rgba(224,92,58,0.07)", line_width=0)
            fig_v.add_trace(go.Scatter(x=vh.index,y=vh.values,line=dict(color="#4a9edd",width=1.5),
                fill="tozeroy",fillcolor="rgba(74,158,221,0.06)",name="VIX",
                hovertemplate="VIX: %{y:.1f}<extra></extra>"))
            for level,label,col in [
                (VIX_CALM,   f"🟢 {VIX_CALM} — calm","#10b981"),
                (VIX_NEUTRAL,f"🟠 {VIX_NEUTRAL} — caution","#f5a623"),
                (VIX_STRESS, f"🔴 {VIX_STRESS} — stress","#e05c3a"),
            ]:
                fig_v.add_hline(y=level,line_color=col,line_dash="dash",line_width=1,
                    annotation_text=label,annotation_font_color=col,
                    annotation_font_size=9,annotation_position="right")
            if d["vix"]:
                fig_v.add_hline(y=d["vix"],line_color="#ffffff",line_dash="dot",line_width=2,
                    annotation_text=f"  👈 NOW {d['vix']:.1f}",annotation_font_color="#ffffff",
                    annotation_font_size=10,annotation_position="right")
            fig_v.update_layout(**PL(height=280,margin=dict(l=10,r=120,t=40,b=10),
                title=dict(text="😨 VIX — Market Fear Gauge  (1 year)",font=dict(color="#ddeeff",size=12))))
            fig_v.update_xaxes(**GRID)
            fig_v.update_yaxes(title="VIX Level",range=[0,55],**GRID)
            st.plotly_chart(fig_v,width='stretch')

        # ── US Yield Curve ─────────────────────────────────────────────────────
        if "t2y" in macro and "t10y" in macro:
            t2  = float(macro["t2y"].dropna().iloc[-1])
            t10 = float(macro["t10y"].dropna().iloc[-1])
            t30 = float(macro["t30y"].dropna().iloc[-1]) if "t30y" in macro else None
            spread=t10-t2
            spread_col="#10b981" if spread>0.5 else "#f5a623" if spread>0 else "#e05c3a"
            spread_emoji="✅" if spread>0.5 else "⚠️" if spread>0 else "🚨"
            spread_words=("Normal — longer bonds pay more than short ones. Economy looks healthy." if spread>0.5
                else "Flattening — the gap is narrow. Worth watching." if spread>0
                else "INVERTED — short rates higher than long. Historically a recession warning.")

            with st.expander("📖 What is the yield curve and why does it matter?", expanded=False):
                st.markdown("""
The **yield curve** plots interest rates (yields) on US government bonds across different maturities — 2 years, 10 years, 30 years.

**Normal curve** (upward sloping): long-term rates > short-term rates. Investors expect growth and demand more compensation for locking up money longer. This is healthy.

**Inverted curve**: short-term rates > long-term rates. This is unusual and has preceded every US recession in modern history, typically 12–18 months later. The 2022 inversion was the deepest in 40 years.

**The 2s10s spread** (10-year minus 2-year yield) is the most-watched version. Positive = normal. Negative = inverted = warning.

**Why it matters for your portfolio:**
- 🏦 Rising rates hurt bond prices (IEF, SCHP)
- 📉 Inverted curves signal tighter financial conditions → bad for growth & EM
- 💵 Steepening curves = economic recovery → good for equities and credit (HYG)
                """)

            maturities=[2,10]+([30] if t30 else [])
            yields=[t2,t10]+([t30] if t30 else [])
            fig_yc=go.Figure()
            fig_yc.add_trace(go.Scatter(x=maturities,y=yields,mode="lines+markers+text",
                line=dict(color="#4a9edd",width=2.5),marker=dict(size=12,color="#4a9edd"),
                text=[f"{v:.2f}%" for v in yields],textposition="top center",
                textfont=dict(color="#ddeeff",size=11),name="🇺🇸 US Yield Curve",
                hovertemplate="%{text}<extra></extra>"))

            # Add Canadian 10Y if available
            if "ca10y" in macro:
                ca10=float(macro["ca10y"].dropna().iloc[-1])
                fig_yc.add_trace(go.Scatter(x=[10],y=[ca10],mode="markers+text",
                    marker=dict(size=12,color="#f5a623",symbol="diamond"),
                    text=[f"🍁 {ca10:.2f}%"],textposition="bottom center",
                    textfont=dict(color="#f5a623",size=10),name="🍁 Canada 10Y"))

            fig_yc.update_layout(**PL(height=240,
                title=dict(text=f"🇺🇸 US Yield Curve  ·  2s10s: {spread:+.2f}%  {spread_emoji}  {spread_words}",
                           font=dict(color=spread_col,size=11))))
            fig_yc.update_xaxes(title="Maturity (years)",tickvals=maturities,
                ticktext=["2Y","10Y"]+["30Y"] if t30 else ["2Y","10Y"],**GRID)
            fig_yc.update_yaxes(title="Yield (%)",ticksuffix="%",**GRID)
            st.plotly_chart(fig_yc,width='stretch')

            # 2s10s rolling spread
            if "t2y" in macro and "t10y" in macro:
                s2=macro["t2y"].dropna(); s10=macro["t10y"].dropna()
                common_idx=s2.index.intersection(s10.index)
                spread_hist=(s10.loc[common_idx]-s2.loc[common_idx]).iloc[-504:]  # ~2yr
                spread_now=spread_hist.iloc[-1]
                sc_now="#10b981" if spread_now>0.5 else "#f5a623" if spread_now>0 else "#e05c3a"
                fig_sp=go.Figure()
                fig_sp.add_hrect(y0=-5,y1=0,fillcolor="rgba(224,92,58,0.08)",line_width=0)
                fig_sp.add_hline(y=0,line_color="#e05c3a",line_dash="dash",line_width=1,
                    annotation_text="  0% — inversion line",annotation_font_color="#e05c3a",annotation_font_size=9)
                fig_sp.add_trace(go.Scatter(x=spread_hist.index,y=spread_hist.values,
                    line=dict(color=sc_now,width=1.5),fill="tozeroy",
                    fillcolor=f"rgba(74,158,221,0.05)",name="2s10s",
                    hovertemplate="%{x|%b %Y}: %{y:+.2f}%<extra></extra>"))
                fig_sp.update_layout(**PL(height=180,
                    title=dict(text=f"📉 2s10s Spread History  (now: {spread_now:+.2f}%)",
                    font=dict(color=sc_now,size=11))))
                fig_sp.update_xaxes(**GRID)
                fig_sp.update_yaxes(title="Spread (%)",ticksuffix="%",**GRID)
                st.plotly_chart(fig_sp,width='stretch')

        # ── CAD/USD ────────────────────────────────────────────────────────────
        if "usdcad" in macro:
            usdcad_s=macro["usdcad"].dropna().iloc[-504:]
            cad_now=float(usdcad_s.iloc[-1])
            cad_1mo=float(usdcad_s.iloc[-22]) if len(usdcad_s)>=22 else cad_now
            cad_chg=cad_now-cad_1mo
            cad_col="#10b981" if cad_chg<0 else "#e05c3a"  # CAD stronger = USD/CAD lower = green
            cad_arrow="↓" if cad_chg<0 else "↑"
            cad_words=("🍁 CAD has strengthened vs USD this month — good for your purchasing power on US-denominated ETFs." if cad_chg<0
                else "🍁 USD has strengthened vs CAD this month — your US-denominated ETFs are worth more in CAD terms.")

            st.markdown(f"""<div style='background:#0a1020;border:1px solid #1e3050;border-radius:6px;
                padding:10px 14px;margin:6px 0 4px 0'>
                <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:6px'>
                <span style='color:#f5a623;font-size:12px;font-weight:700'>🍁 USD/CAD Exchange Rate</span>
                <span style='color:#ddeeff;font-size:14px;font-weight:700'>{cad_now:.4f}
                <span style='color:{cad_col};font-size:11px;margin-left:8px'>{cad_arrow}{abs(cad_chg):.4f} vs 1mo ago</span>
                </span></div>
                <div style='font-size:10px;color:#4a6fa5;line-height:1.5'>{cad_words}<br>
                As a Canadian investor holding USD ETFs, a weaker CAD (higher USD/CAD) means your portfolio
                is worth <i>more</i> in Canadian dollars even without any price gains. It cuts both ways on repatriation.</div>
            </div>""",unsafe_allow_html=True)

            fig_cad=go.Figure()
            fig_cad.add_trace(go.Scatter(x=usdcad_s.index,y=usdcad_s.values,
                line=dict(color="#f5a623",width=1.5),fill="tozeroy",
                fillcolor="rgba(245,167,35,0.05)",name="USD/CAD",
                hovertemplate="%{x|%b %Y}: %{y:.4f}<extra></extra>"))
            fig_cad.add_hline(y=cad_now,line_color="#ffffff",line_dash="dot",line_width=1,
                annotation_text=f"  Now: {cad_now:.4f}",annotation_font_color="#ffffff",annotation_font_size=9)
            fig_cad.update_layout(**PL(height=180,
                title=dict(text=f"🍁 USD/CAD  ·  1 = {cad_now:.4f}  ·  Higher = weaker CAD",
                font=dict(color="#f5a623",size=11))))
            fig_cad.update_xaxes(**GRID)
            fig_cad.update_yaxes(title="USD/CAD",**GRID)
            st.plotly_chart(fig_cad,width='stretch')

        # ── Macro panel (FRED) ────────────────────────────────────────────────
        if macro:
            st.markdown("""<div style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:8px'>
                🌍 MACRO ENVIRONMENT  (via US Federal Reserve — FRED)</div>""",unsafe_allow_html=True)

            with st.expander("📖 What is FRED and why are these indicators relevant to you?", expanded=False):
                st.markdown("""
**FRED** (Federal Reserve Economic Data) is a free database from the St. Louis Fed with 800,000+ economic time series.
These three indicators give you a real-time read on the macro backdrop your portfolio is operating in:

| Indicator | What it measures | Why it matters |
|-----------|-----------------|---------------|
| 💥 HYG OAS Spread | Extra yield on junk bonds vs Treasuries | Fear gauge for corporate credit — spikes in crises |
| 🔥 10Y Breakeven | Bond market's 10-year inflation forecast | Drives Fed policy expectations |
| 🏦 10Y Real Yield | Yield after inflation — your real cost of capital | Rising = tightening = headwind for growth & EM |

**As a Canadian investor**, these are US indicators but they drive global risk conditions.
When US credit spreads spike, EM and HYG sell off everywhere — including your portfolio.
                """)

            macro_config=[
                ("hyg_oas",   "💥 Credit Stress — HYG Spread (bps)", "bps", 100,
                 "Extra yield junk bonds pay over safe Treasuries. "
                 "<b style='color:#e05c3a'>Rising = companies seen as riskier = bad for HYG and risk assets.</b> "
                 "Below 300bps = calm. 400+ = stress. 600+ = crisis."),
                ("breakeven", "🔥 Inflation Expectations — 10Y Breakeven (%)", "%", 1,
                 "What the bond market thinks inflation will average over 10 years. "
                 "<b style='color:#f5a623'>Above 2.5% = inflation worry. The Fed targets 2%.</b> "
                 "Drives Fed rate decisions, which ripple through everything."),
                ("real_yield","🏦 Real Yield — 10Y TIPS (%)", "%", 1,
                 "Yield after inflation. What you actually earn in purchasing power. "
                 "<b style='color:#e05c3a'>Rising = tighter financial conditions = headwind for growth stocks and EM.</b> "
                 "The 2022 spike crushed bonds and growth stocks simultaneously."),
            ]
            for key,label,unit,scale,explanation in macro_config:
                if key in macro:
                    sd=macro[key].dropna()
                    if len(sd)>=2:
                        cur=float(sd.iloc[-1])*scale
                        chg=float((sd.iloc[-1]-sd.iloc[-22])*scale) if len(sd)>=22 else 0.0
                        chg_col=("#e05c3a" if (chg>0 and key in ["hyg_oas","real_yield"])
                                 else "#10b981" if chg>0 else "#e05c3a")
                        chg_arrow="↑" if chg>0 else "↓"

                        # Mini sparkline
                        spark=sd.iloc[-63:]*scale  # ~3 months
                        fig_sp=go.Figure(go.Scatter(
                            x=spark.index,y=spark.values,
                            line=dict(color="#4a9edd",width=1),
                            fill="tozeroy",fillcolor="rgba(74,158,221,0.06)",
                            hovertemplate="%{x|%b %d}: %{y:.2f}<extra></extra>"))
                        fig_sp.update_layout(**PL(height=80,
                            margin=dict(l=0,r=0,t=0,b=0),
                            paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)"))
                        fig_sp.update_xaxes(visible=False,**GRID)
                        fig_sp.update_yaxes(visible=False,**GRID)

                        st.markdown(f"""<div style='background:#0d1525;border:1px solid #162030;
                            border-radius:8px;padding:10px 14px;margin-bottom:8px'>
                            <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px'>
                            <span style='color:#ddeeff;font-size:12px;font-weight:700'>{label}</span>
                            <span style='color:#ddeeff;font-size:15px;font-weight:700'>{cur:.1f}{unit}
                            <span style='color:{chg_col};font-size:11px;margin-left:8px'>{chg_arrow}{abs(chg):.1f} vs 1mo ago</span>
                            </span></div>
                            <div style='font-size:10px;color:#4a6fa5;line-height:1.5;margin-bottom:6px'>{explanation}</div>
                        </div>""",unsafe_allow_html=True)
                        st.plotly_chart(fig_sp,width='stretch')

        # ── Carry signal ───────────────────────────────────────────────────────
        if not d["carry_df"].empty:
            st.markdown("""<div style='font-size:10px;color:#a855f7;letter-spacing:2px;margin-top:10px'>
                💰 CARRY SIGNAL</div>""",unsafe_allow_html=True)
            with st.expander("📖 What is carry?", expanded=False):
                st.markdown("""
**Carry** is the yield you earn just for *holding* an asset, regardless of price moves.

For **HYG** (high-yield bonds), carry is the credit spread — the extra yield junk bonds pay over Treasuries.

- ✅ **Positive carry** = the yield compensates for the risk. Worth holding.
- ❌ **Negative carry** = the yield is too thin. You're not being paid enough to take the risk.

Think of it like a landlord calculating whether rent covers the mortgage.
If the rent (yield) doesn't cover the cost and risk, you'd rather not own the property.
                """)
            for name,row in d["carry_df"].iterrows():
                sig=int(row["signal"]) if pd.notna(row["signal"]) else 0
                sc_c="#10b981" if sig>0 else "#e05c3a" if sig<0 else "#4a6fa5"
                sig_text=("✅ Carry is attractive — yield compensates for risk" if sig>0
                          else "❌ Carry is unattractive — spread too thin" if sig<0
                          else "➡️ Neutral — borderline carry")
                st.markdown(f"""<div style='background:#0d1525;border:1px solid #162030;border-radius:6px;
                    padding:10px 14px;margin-bottom:6px'>
                    <div style='display:flex;justify-content:space-between;align-items:center'>
                    <span style='color:#8da0bc;font-size:12px;font-weight:600'>{name}</span>
                    <span style='color:{sc_c};font-size:12px;font-weight:700'>
                    {"▲ POSITIVE CARRY" if sig>0 else "▼ NEGATIVE CARRY" if sig<0 else "── NEUTRAL"}</span></div>
                    <div style='font-size:11px;color:#4a6fa5;margin-top:4px'>{sig_text}</div>
                </div>""",unsafe_allow_html=True)

    # ── News feed ──────────────────────────────────────────────────────────────
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:16px'>📰 NEWS FEED — select a ticker</p>",
                unsafe_allow_html=True)
    sel=st.selectbox("Ticker",TICKERS_LIST,key="news_ticker")
    with st.spinner(f"Fetching {sel} news..."):
        news_data=load_news()
    articles=news_data.get(sel,[])
    if not articles:
        st.markdown("<div style='color:#4a6fa5;font-size:11px;padding:8px'>No recent news found for this ticker.</div>",unsafe_allow_html=True)
    for art in articles:
        title=art.get("title",""); pub=art.get("publisher",""); link=art.get("link","#")
        pt=art.get("published","")
        if isinstance(pt,(int,float)) and pt:
            try: pt=datetime.fromtimestamp(pt).strftime("%b %d, %Y %H:%M")
            except: pt=str(pt)
        if title:
            st.markdown(f"""<div style='background:#0d1525;border:1px solid #162030;border-radius:6px;
                padding:10px 14px;margin-bottom:6px'>
                <a href='{link}' target='_blank' style='color:#4a9edd;font-size:12px;font-weight:600;
                text-decoration:none'>{title}</a>
                <div style='color:#4a6fa5;font-size:10px;margin-top:4px'>📰 {pub}  ·  🕐 {pt}</div>
            </div>""",unsafe_allow_html=True)

# ══ TAB 3: PERFORMANCE ═══════════════════════════════════════════════════════
with tab3:
    fig=go.Figure()
    port_cum=cum(port_ret)
    fig.add_trace(go.Scatter(x=port_cum.index,y=port_cum.values,name="Portfolio",
        line=dict(color="#4a9edd",width=2.5),fill="tozeroy",fillcolor="rgba(74,158,221,0.06)"))
    for bname,bcol in BENCHMARKS.items():
        if bname in prices.columns:
            br=prices[bname].pct_change(fill_method=None).dropna()
            idx=port_cum.index.intersection(br.index)
            fig.add_trace(go.Scatter(x=idx,y=cum(br.loc[idx]).values,name=bname,
                line=dict(color=bcol,width=1.5,dash="dot")))
    fig.update_layout(**PL(height=360,hovermode="x unified",
        title=dict(text="📈 Cumulative Returns — Portfolio vs SPY vs VTI",
                   font=dict(color="#ddeeff",size=13))))
    fig.update_xaxes(**GRID); fig.update_yaxes(**GRID)
    st.plotly_chart(fig,width='stretch')

    ps=ann_stats(port_ret)
    ncols=1+len([b for b in BENCHMARKS if b in prices.columns])+1
    sc=st.columns(ncols)
    sc[0].markdown("<div style='font-size:9px;color:#4a6fa5;letter-spacing:1px'>PORTFOLIO</div>",unsafe_allow_html=True)
    sc[0].metric("Ann Ret",  f"{ps['ret']:.1f}%")
    sc[0].metric("Vol",      f"{ps['vol']:.1f}%")
    sc[0].metric("Sharpe",   f"{ps['sr']:.2f}")
    sc[0].metric("Sortino",  f"{ps['sortino']:.2f}")
    sc[0].metric("Calmar",   f"{ps['calmar']:.2f}")
    sc[0].metric("Max DD",   f"{ps['dd']:.1f}%")
    ci=1
    for bname,bcol in BENCHMARKS.items():
        if bname in prices.columns:
            bs=ann_stats(prices[bname].pct_change(fill_method=None).dropna())
            sc[ci].markdown(f"<div style='font-size:9px;color:{bcol};letter-spacing:1px'>{bname}</div>",unsafe_allow_html=True)
            sc[ci].metric("Ann Ret", f"{bs['ret']:.1f}%", f"{ps['ret']-bs['ret']:+.1f}%")
            sc[ci].metric("Vol",     f"{bs['vol']:.1f}%")
            sc[ci].metric("Sharpe",  f"{bs['sr']:.2f}",   f"{ps['sr']-bs['sr']:+.2f}")
            sc[ci].metric("Sortino", f"{bs['sortino']:.2f}")
            sc[ci].metric("Calmar",  f"{bs['calmar']:.2f}")
            sc[ci].metric("Max DD",  f"{bs['dd']:.1f}%")
            ci+=1
    sc[ci].markdown("<div style='font-size:9px;color:#4a6fa5;letter-spacing:1px'>PERIODS</div>",unsafe_allow_html=True)
    sc[ci].metric("MTD",f"{dp['mtd']:+.2f}%",delta_color="normal")
    sc[ci].metric("YTD",f"{dp['ytd']:+.2f}%",delta_color="normal")
    sc[ci].metric("1W", f"{dp['week']:+.2f}%",delta_color="normal")
    sc[ci].metric("1D", f"{dp['day']:+.2f}%", delta_color="normal")

    # Drawdown — portfolio + benchmarks overlaid
    fig_dd=go.Figure()
    dd=(port_cum/port_cum.cummax()-1)
    fig_dd.add_trace(go.Scatter(x=dd.index,y=dd.values*100,fill="tozeroy",
        fillcolor="rgba(224,92,58,0.10)",line=dict(color="#e05c3a",width=1.5),name="Portfolio"))
    for bname,bcol in BENCHMARKS.items():
        if bname in prices.columns:
            br=prices[bname].pct_change(fill_method=None).dropna()
            bc=cum(br.reindex(port_cum.index).dropna())
            bdd=(bc/bc.cummax()-1)*100
            fig_dd.add_trace(go.Scatter(x=bdd.index,y=bdd.values,
                line=dict(color=bcol,width=1,dash="dot"),name=bname))
    fig_dd.update_layout(**PL(height=220,hovermode="x unified",
        title=dict(text="Drawdown — Portfolio vs Benchmarks",font=dict(color="#ddeeff",size=11))))
    fig_dd.update_xaxes(**GRID); fig_dd.update_yaxes(ticksuffix="%",**GRID)
    st.plotly_chart(fig_dd,width='stretch')

    # ── Daily attribution table ───────────────────────────────────────────────
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:8px'>"
                "📋 DAILY ATTRIBUTION — WHAT IS DRIVING PERFORMANCE</p>",unsafe_allow_html=True)

    attr_period = st.radio("Attribution window",["1D","3D","1W","2W","1M"],horizontal=True,index=0,key="attr_period")
    n_attr = {"1D":1,"3D":3,"1W":5,"2W":10,"1M":21}[attr_period]

    av_attr = [t for t in TICKERS_LIST if t in prices.columns]
    wattr   = target_w.reindex(av_attr).fillna(0); wattr = wattr/wattr.sum()
    px_attr = prices[av_attr].pct_change(fill_method=None).dropna().iloc[-n_attr:]

    # Position-level attribution
    attr_rows = []
    for t in av_attr:
        pos_ret  = ((1+px_attr[t]).prod()-1)*100
        weighted = pos_ret * float(wattr[t])
        mom_sig  = float(d["trend_df"].loc[t,"signal"]) if not d["trend_df"].empty and t in d["trend_df"].index else 0
        z_score  = float(d["trend_df"].loc[t,"z_score"]) if not d["trend_df"].empty and t in d["trend_df"].index else None
        tilt     = d["tilts"].get(t, 0)
        sig_icon = "▲" if mom_sig > 0 else "▼" if mom_sig < 0 else "──"
        attr_rows.append({
            "Ticker":       t,
            "Name":         TICKERS[t]["label"],
            "Weight":       f"{float(wattr[t]):.1%}",
            f"Ret ({attr_period})": f"{pos_ret:+.2f}%",
            "Contribution": f"{weighted:+.3f}%",
            "Signal":       sig_icon,
            "Z-Score":      f"{z_score:+.2f}" if z_score is not None else "—",
            "Regime Tilt":  f"{tilt:+.0%}" if tilt else "—",
            "Sleeve":       TICKERS[t]["sleeve"],
        })
    attr_df = pd.DataFrame(attr_rows).sort_values("Contribution",
        key=lambda x: x.str.replace("+","").str.replace("%","").astype(float), ascending=False)

    # Colour-code contribution column
    def style_contrib(val):
        try:
            v = float(val.replace("+","").replace("%",""))
            if v > 0:   return "color: #10b981; font-weight: bold"
            elif v < 0: return "color: #e05c3a; font-weight: bold"
        except: pass
        return ""

    st.dataframe(attr_df.set_index("Ticker").style.map(style_contrib, subset=["Contribution"]),
                 width='stretch', height=340)

    # Attribution waterfall chart
    attr_vals = pd.Series(
        {r["Ticker"]: float(r["Contribution"].replace("+","").replace("%","")) for r in attr_rows}
    ).sort_values()

    fig_attr = go.Figure(go.Bar(
        x=attr_vals.values, y=attr_vals.index, orientation="h",
        marker_color=["#10b981" if v>=0 else "#e05c3a" for v in attr_vals.values],
        marker_line_width=0,
        text=[f"{v:+.3f}%" for v in attr_vals.values],
        textposition="outside", textfont=dict(color="#8da0bc",size=10),
        hovertemplate="<b>%{y}</b>: %{x:+.3f}%<extra></extra>"))
    fig_attr.add_vline(x=0,line_color="#4a6fa5",line_dash="dot")
    # Total portfolio return line
    total = attr_vals.sum()
    fig_attr.add_vline(x=total,line_color="#ddeeff",line_dash="dash",line_width=1.5,
        annotation_text=f" Total {total:+.2f}%",annotation_font_color="#ddeeff",annotation_font_size=10)

    # Add signal context as annotation on each bar
    fig_attr.update_layout(**PL(height=340,
        title=dict(text=f"Position Contribution ({attr_period})  ·  dashed = portfolio total",
                   font=dict(color="#ddeeff",size=12))))
    fig_attr.update_xaxes(title="Weighted Contribution (%)",ticksuffix="%",**GRID)
    fig_attr.update_yaxes(**GRID)
    st.plotly_chart(fig_attr,width='stretch')

    # Signal vs performance table — ties attribution to signals
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:6px'>"
                "🔗 SIGNAL vs PERFORMANCE — ARE SIGNALS WORKING?</p>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-bottom:6px'>"
                "Green = signal and return agree (signal helped). "
                "Red = signal and return disagree (signal hurt or no signal, still moved).</div>",
                unsafe_allow_html=True)
    sig_rows = []
    for r in attr_rows:
        t = r["Ticker"]
        ret_val = float(r[f"Ret ({attr_period})"].replace("+","").replace("%",""))
        sig_val = float(d["trend_df"].loc[t,"signal"]) if not d["trend_df"].empty and t in d["trend_df"].index else 0
        if sig_val > 0 and ret_val > 0:   agreement = "✅ Long + Up"
        elif sig_val < 0 and ret_val < 0: agreement = "✅ Short + Down"
        elif sig_val > 0 and ret_val < 0: agreement = "❌ Long + Down"
        elif sig_val < 0 and ret_val > 0: agreement = "❌ Short + Up"
        else:                              agreement = "── Neutral"
        sig_rows.append({"Ticker":t,f"Ret ({attr_period})":r[f"Ret ({attr_period})"],
                         "Signal":r["Signal"],"Agreement":agreement,
                         "Z-Score":r["Z-Score"],"Contribution":r["Contribution"]})
    sig_df = pd.DataFrame(sig_rows).set_index("Ticker")
    st.dataframe(sig_df, width='stretch', height=300)

# ══ TAB 4: VS BENCHMARK ══════════════════════════════════════════════════════
with tab4:
    bench_rets={b:prices[b].pct_change(fill_method=None).dropna() for b in BENCHMARKS if b in prices.columns}
    if not bench_rets:
        st.warning("Run `python3 -m data.fetcher` to load SPY and VTI.")
    else:
        min_date=port_ret.index[0].date(); max_date=port_ret.index[-1].date()
        dc1,dc2=st.columns(2)
        with dc1: start_date=st.date_input("📅 From",value=min_date,min_value=min_date,max_value=max_date,key="bs")
        with dc2: end_date  =st.date_input("📅 To",  value=max_date,min_value=min_date,max_value=max_date,key="be")

        pr_f ={b:r[(r.index.date>=start_date)&(r.index.date<=end_date)] for b,r in bench_rets.items()}
        port_f=port_ret[(port_ret.index.date>=start_date)&(port_ret.index.date<=end_date)]

        fig_all=go.Figure()
        if len(port_f)>0:
            pc=cum(port_f)
            fig_all.add_trace(go.Scatter(x=pc.index,y=pc.values,name="Portfolio",
                line=dict(color="#4a9edd",width=2.5)))
        for bname,bret in pr_f.items():
            if len(bret)>0:
                idx=pc.index.intersection(bret.index) if len(port_f)>0 else bret.index
                fig_all.add_trace(go.Scatter(x=idx,y=cum(bret.loc[idx]).values,name=bname,
                    line=dict(color=BENCHMARKS[bname],width=1.5,dash="dot")))
        fig_all.update_layout(**PL(height=300,hovermode="x unified",
            title=dict(text="⚔️ Portfolio vs SPY vs VTI",font=dict(color="#ddeeff",size=12))))
        fig_all.update_xaxes(**GRID); fig_all.update_yaxes(**GRID)
        st.plotly_chart(fig_all,width='stretch')

        fig_rel=go.Figure()
        for bname,bret in pr_f.items():
            idx=port_f.index.intersection(bret.index)
            if len(idx)>20:
                excess=port_f.loc[idx]-bret.loc[idx]
                roll_n=min(63,max(5,len(idx)//3))
                rolling=excess.rolling(roll_n).sum()*100
                rhex=BENCHMARKS[bname]
                fig_rel.add_trace(go.Scatter(x=rolling.index,y=rolling.values,name=f"vs {bname}",
                    line=dict(color=rhex,width=2),fill="tozeroy",
                    fillcolor=f"rgba({int(rhex[1:3],16)},{int(rhex[3:5],16)},{int(rhex[5:7],16)},0.06)",
                    hovertemplate=f"vs {bname}: %{{y:+.1f}}%<extra></extra>"))
        fig_rel.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
        fig_rel.update_layout(**PL(height=240,hovermode="x unified",
            title=dict(text="🔄 Rolling Excess Return",font=dict(color="#ddeeff",size=12))))
        fig_rel.update_xaxes(**GRID); fig_rel.update_yaxes(ticksuffix="%",**GRID)
        st.plotly_chart(fig_rel,width='stretch')

        fig_exc=go.Figure()
        for bname,bret in pr_f.items():
            idx=port_f.index.intersection(bret.index)
            if len(idx)>1:
                fig_exc.add_trace(go.Scatter(x=idx,y=(cum(port_f.loc[idx]-bret.loc[idx])-1)*100,
                    name=f"vs {bname}",line=dict(color=BENCHMARKS[bname],width=2)))
        fig_exc.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
        fig_exc.update_layout(**PL(height=220,hovermode="x unified",
            title=dict(text="📊 Cumulative Excess Return",font=dict(color="#ddeeff",size=12))))
        fig_exc.update_xaxes(**GRID); fig_exc.update_yaxes(ticksuffix="%",**GRID)
        st.plotly_chart(fig_exc,width='stretch')

        ps_f=ann_stats(port_f)
        rows=[{"":"Portfolio","Ann Ret":f"{ps_f['ret']:.1f}%","Vol":f"{ps_f['vol']:.1f}%",
               "Sharpe":f"{ps_f['sr']:.2f}","Sortino":f"{ps_f['sortino']:.2f}",
               "Max DD":f"{ps_f['dd']:.1f}%","Track Err":"—","Info Ratio":"—"}]
        for bname,bret in pr_f.items():
            bs=ann_stats(bret); common=port_f.index.intersection(bret.index)
            te=(port_f.loc[common]-bret.loc[common]).std()*np.sqrt(252)*100 if len(common)>1 else 0
            ir=(ps_f['ret']-bs['ret'])/te if te>0 else 0
            rows.append({"":bname,"Ann Ret":f"{bs['ret']:.1f}%","Vol":f"{bs['vol']:.1f}%",
                "Sharpe":f"{bs['sr']:.2f}","Sortino":f"{bs['sortino']:.2f}",
                "Max DD":f"{bs['dd']:.1f}%","Track Err":f"{te:.1f}%","Info Ratio":f"{ir:.2f}"})
        st.dataframe(pd.DataFrame(rows).set_index(""),width='stretch')

# ══ TAB 5: RETURN ANALYZER ═══════════════════════════════════════════════════
with tab5:
    av=[t for t in TICKERS_LIST if t in prices.columns]
    w_n=target_w.reindex(av).fillna(0); w_n=w_n/w_n.sum()
    px_r=prices[av].pct_change(fill_method=None).dropna()
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px'>🔬 RETURN CONTRIBUTION</p>",
                unsafe_allow_html=True)
    period=st.radio("Period",["1D","1W","1M","3M","1Y","All"],horizontal=True,index=3)
    n_map={"1D":1,"1W":5,"1M":21,"3M":63,"1Y":252,"All":len(px_r)}
    n=n_map[period]; sl=px_r.iloc[-n:]
    contrib={t:((1+sl[t]).prod()-1)*float(w_n[t])*100 for t in av}
    cs=pd.Series(contrib).sort_values()
    fig_c=go.Figure(go.Bar(x=cs.values,y=cs.index,orientation="h",
        marker_color=["#10b981" if v>=0 else "#e05c3a" for v in cs.values],marker_line_width=0,
        text=[f"{v:+.2f}%" for v in cs.values],textposition="outside",textfont=dict(color="#8da0bc",size=10)))
    fig_c.add_vline(x=0,line_color="#4a6fa5",line_dash="dot")
    fig_c.update_layout(**PL(height=320,title=dict(text=f"Weighted Return Contribution ({period})",
        font=dict(color="#ddeeff",size=12))))
    fig_c.update_xaxes(ticksuffix="%",**GRID); fig_c.update_yaxes(**GRID)
    st.plotly_chart(fig_c,width='stretch')

    c1,c2=st.columns(2)
    sl_c={"CORE":0.0,"FACTOR":0.0}
    for t in av: sl_c[TICKERS[t]["sleeve"].upper()]+=contrib.get(t,0)
    with c1:
        fig_sl=go.Figure(go.Bar(x=list(sl_c.keys()),y=list(sl_c.values()),
            marker_color=[SLEEVE_COL["core"],SLEEVE_COL["factor"]],marker_line_width=0,
            text=[f"{v:+.2f}%" for v in sl_c.values()],textposition="outside",
            textfont=dict(color="#8da0bc",size=11)))
        fig_sl.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
        fig_sl.update_layout(**PL(height=260,title=dict(text="🧩 Sleeve Attribution",font=dict(color="#ddeeff",size=11))))
        fig_sl.update_xaxes(**GRID); fig_sl.update_yaxes(ticksuffix="%",**GRID)
        st.plotly_chart(fig_sl,width='stretch')
    with c2:
        bc={"Portfolio":sum(contrib.values())}
        for bname in BENCHMARKS:
            if bname in prices.columns:
                bc[bname]=((1+prices[bname].pct_change(fill_method=None).dropna().iloc[-n:]).prod()-1)*100
        fig_bc=go.Figure(go.Bar(x=list(bc.keys()),y=list(bc.values()),
            marker_color=["#4a9edd"]+[BENCHMARKS[b] for b in bc if b!="Portfolio"],
            marker_line_width=0,text=[f"{v:+.1f}%" for v in bc.values()],textposition="outside",
            textfont=dict(color="#8da0bc",size=11)))
        fig_bc.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
        fig_bc.update_layout(**PL(height=260,title=dict(text=f"vs Benchmarks ({period})",font=dict(color="#ddeeff",size=11))))
        fig_bc.update_xaxes(**GRID); fig_bc.update_yaxes(ticksuffix="%",**GRID)
        st.plotly_chart(fig_bc,width='stretch')

    corr=px_r.iloc[-90:].corr()
    fig_h=go.Figure(go.Heatmap(z=corr.values,x=corr.columns,y=corr.index,
        colorscale=[[0,"#e05c3a"],[0.5,"#0d1525"],[1,"#4a9edd"]],zmin=-1,zmax=1,
        text=corr.round(2).values,texttemplate="%{text}",textfont=dict(size=9,color="#ddeeff"),
        hovertemplate="%{y} / %{x}: %{z:.2f}<extra></extra>"))
    fig_h.update_layout(**PL(height=360,title=dict(text="🔗 Correlation Matrix (90d)",
        font=dict(color="#ddeeff",size=12))))
    st.plotly_chart(fig_h,width='stretch')

# ══ TAB 6: STRESS TEST ═══════════════════════════════════════════════════════
with tab6:
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px'>💥 HISTORICAL STRESS TEST</p>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-bottom:12px'>Replays your current "
                "target weights through historical crisis periods using actual price data. Shows what "
                "you <i>would have</i> lost — and how you compared to SPY and VTI.</div>",
                unsafe_allow_html=True)

    crisis_results=[]
    for crisis_name,(cstart,cend,spy_ref) in CRISES.items():
        try:
            cs_idx=prices.index[(prices.index>=cstart)&(prices.index<=cend)]
            if len(cs_idx)<5: continue
            cs_px=prices.loc[cs_idx]
            # Portfolio return
            av_c=[t for t in TICKERS_LIST if t in cs_px.columns]
            wn_c=target_w.reindex(av_c).fillna(0); wn_c=wn_c/wn_c.sum()
            pr_c=cs_px[av_c].pct_change(fill_method=None).dropna().dot(wn_c)
            port_cr=((1+pr_c).prod()-1)*100
            # Drawdown during
            cum_c=(1+pr_c).cumprod()
            dd_c=(cum_c/cum_c.cummax()-1).min()*100
            # Benchmarks
            bench_cr={}
            for bname in BENCHMARKS:
                if bname in cs_px.columns:
                    br_c=cs_px[bname].pct_change(fill_method=None).dropna()
                    bench_cr[bname]=((1+br_c).prod()-1)*100
            row={"Episode":crisis_name,"Period":f"{cstart[:7]} → {cend[:7]}",
                 "Reference":spy_ref,"Duration":f"{len(cs_idx)}d",
                 "Portfolio":round(port_cr,1),"Max DD":round(dd_c,1)}
            for bname,br in bench_cr.items():
                row[bname]=round(br,1)
                row[f"vs {bname}"]=round(port_cr-br,1)
            crisis_results.append(row)
        except:
            continue

    if crisis_results:
        cdf=pd.DataFrame(crisis_results)

        # Summary chart
        fig_st=go.Figure()
        x_labels=cdf["Episode"].tolist()
        fig_st.add_trace(go.Bar(name="Portfolio",x=x_labels,y=cdf["Portfolio"].tolist(),
            marker_color=["#10b981" if v>=0 else "#4a9edd" for v in cdf["Portfolio"]],
            marker_line_width=0,text=[f"{v:+.1f}%" for v in cdf["Portfolio"]],
            textposition="outside",textfont=dict(color="#8da0bc",size=9)))
        for bname,bcol in BENCHMARKS.items():
            if bname in cdf.columns:
                fig_st.add_trace(go.Bar(name=bname,x=x_labels,y=cdf[bname].tolist(),
                    marker_color=bcol,marker_opacity=0.7,marker_line_width=0,
                    text=[f"{v:+.1f}%" for v in cdf[bname]],textposition="outside",
                    textfont=dict(color="#8da0bc",size=9)))
        fig_st.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
        fig_st.update_layout(**PL(height=380,barmode="group",hovermode="x unified",
            title=dict(text="💥 Portfolio vs Benchmarks Through Historical Crises",
                       font=dict(color="#ddeeff",size=13))))
        fig_st.update_xaxes(**GRID); fig_st.update_yaxes(title="Total Return (%)",ticksuffix="%",**GRID)
        st.plotly_chart(fig_st,width='stretch')

        # Relative outperformance bar
        fig_rel=go.Figure()
        for bname in BENCHMARKS:
            col_name=f"vs {bname}"
            if col_name in cdf.columns:
                fig_rel.add_trace(go.Bar(name=f"vs {bname}",x=x_labels,y=cdf[col_name].tolist(),
                    marker_color=[BENCHMARKS[bname] if v>=0 else "#e05c3a" for v in cdf[col_name]],
                    marker_line_width=0,opacity=0.85,
                    text=[f"{v:+.1f}%" for v in cdf[col_name]],textposition="outside",
                    textfont=dict(color="#8da0bc",size=9)))
        fig_rel.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
        fig_rel.update_layout(**PL(height=280,barmode="group",
            title=dict(text="Portfolio Outperformance vs Each Benchmark (positive = portfolio won)",
                       font=dict(color="#ddeeff",size=11))))
        fig_rel.update_xaxes(**GRID); fig_rel.update_yaxes(ticksuffix="%",**GRID)
        st.plotly_chart(fig_rel,width='stretch')

        # Detail table
        display_cols=["Episode","Period","Reference","Duration","Portfolio","Max DD"]+\
                     [b for b in BENCHMARKS if b in cdf.columns]+\
                     [f"vs {b}" for b in BENCHMARKS if f"vs {b}" in cdf.columns]
        fmt_cdf=cdf[display_cols].copy()
        for col in ["Portfolio","Max DD"]+[b for b in BENCHMARKS if b in cdf.columns]+\
                   [f"vs {b}" for b in BENCHMARKS if f"vs {b}" in cdf.columns]:
            fmt_cdf[col]=fmt_cdf[col].map(lambda x:f"{x:+.1f}%")
        st.dataframe(fmt_cdf.set_index("Episode"),width='stretch',height=260)

        # Scenario builder
        st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:16px'>"
                    "🔧 CUSTOM SCENARIO</p>",unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-bottom:8px'>"
                    "Apply hypothetical shocks to each asset class and see portfolio impact.</div>",
                    unsafe_allow_html=True)
        scen_cols=st.columns(4)
        shocks={}
        defaults={"SPY":-20,"EFA":-18,"EEM":-25,"IEF":5,"SCHP":3,"HYG":-12,
                  "QMOM":-15,"QUAL":-10,"AVUV":-22,"KMLM":8,"DBMF":6}
        for i,t in enumerate(TICKERS_LIST):
            with scen_cols[i%4]:
                shocks[t]=st.number_input(f"{t} shock %",value=defaults.get(t,0),
                                          min_value=-80,max_value=80,step=1,key=f"shock_{t}")
        shock_impact=sum(shocks[t]/100*float(target_w.get(t,0)) for t in TICKERS_LIST)*100
        spy_shock=shocks.get("SPY",0)
        col_sc=["#10b981" if shock_impact>=0 else "#e05c3a"]
        st.markdown(f"""<div style='background:#0d1525;border-left:4px solid {col_sc[0]};
            border-radius:6px;padding:14px 18px;margin-top:8px'>
            <div style='font-size:11px;color:#8da0bc'>Scenario Portfolio Impact</div>
            <div style='font-size:28px;font-weight:700;color:{col_sc[0]}'>{shock_impact:+.2f}%</div>
            <div style='font-size:10px;color:#4a6fa5;margin-top:4px'>
            SPY assumption: {spy_shock:+.0f}%  ·  Portfolio vs SPY: {shock_impact-spy_shock:+.2f}%</div>
        </div>""",unsafe_allow_html=True)
    else:
        st.info("Not enough historical price data to run stress tests. Run `python3 -m data.fetcher` with a longer period.")

# ══ TAB 7: MONTE CARLO ═══════════════════════════════════════════════════════
with tab7:
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px'>🎲 MONTE CARLO SIMULATION</p>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-bottom:10px'>Forward simulation using "
                "AQR 2026 CMA expected returns + historical volatility. Each path is a possible future. "
                "The fan shows the 10th–90th percentile range of outcomes.</div>",unsafe_allow_html=True)

    mc1,mc2,mc3,mc4=st.columns(4)
    with mc1: mc_value   = st.number_input("Starting Value ($)",value=100_000,step=10_000,format="%d",key="mc_val")
    with mc2: mc_years   = st.slider("Years",1,30,10,key="mc_yr")
    with mc3: mc_contrib = st.number_input("Monthly Contribution ($)",value=0,step=500,format="%d",key="mc_contrib")
    with mc4: mc_n       = st.select_slider("Simulations",options=[500,1000,2000,5000],value=1000,key="mc_n")

    # Build portfolio params from AQR CMAs
    av_mc=[t for t in TICKERS_LIST if t in prices.columns]
    wn_mc=target_w.reindex(av_mc).fillna(0); wn_mc=wn_mc/wn_mc.sum()
    mu_daily  = sum(EXPECTED_RETURNS[t]/100/252*float(wn_mc.get(t,0)) for t in av_mc)
    # Use historical covariance for simulation
    px_mc=prices[av_mc].pct_change(fill_method=None).dropna().iloc[-756:]  # 3yr history
    cov_mc=(px_mc.cov()*252).values
    w_arr=wn_mc.values if hasattr(wn_mc,'values') else np.array([float(wn_mc.get(t,0)) for t in av_mc])
    port_vol_ann=np.sqrt(w_arr@cov_mc@w_arr)
    port_vol_daily=port_vol_ann/np.sqrt(252)
    # Override mean with AQR expected return
    er_ann=s["expected_return"]/100
    er_daily=(1+er_ann)**(1/252)-1

    trading_days=mc_years*252
    np.random.seed(123)
    all_paths=np.zeros((mc_n, trading_days+1))
    all_paths[:,0]=mc_value

    for sim in range(mc_n):
        daily_rets=np.random.normal(er_daily, port_vol_daily, trading_days)
        val=mc_value
        for i,dr in enumerate(daily_rets):
            val=val*(1+dr)+(mc_contrib/21)  # monthly contrib spread daily
            all_paths[sim,i+1]=val

    # Percentiles
    pct10 =np.percentile(all_paths,10, axis=0)
    pct25 =np.percentile(all_paths,25, axis=0)
    pct50 =np.percentile(all_paths,50, axis=0)
    pct75 =np.percentile(all_paths,75, axis=0)
    pct90 =np.percentile(all_paths,90, axis=0)
    x_yrs =np.linspace(0,mc_years,trading_days+1)

    fig_mc=go.Figure()
    # Fan
    fig_mc.add_trace(go.Scatter(x=np.concatenate([x_yrs,x_yrs[::-1]]),
        y=np.concatenate([pct90,pct10[::-1]]),fill="toself",
        fillcolor="rgba(74,158,221,0.08)",line=dict(width=0),name="10th–90th pct",showlegend=True))
    fig_mc.add_trace(go.Scatter(x=np.concatenate([x_yrs,x_yrs[::-1]]),
        y=np.concatenate([pct75,pct25[::-1]]),fill="toself",
        fillcolor="rgba(74,158,221,0.15)",line=dict(width=0),name="25th–75th pct",showlegend=True))
    # Percentile lines
    for pct_vals,pct_name,col,width in [
        (pct90,"90th pct","#10b981",1.5),(pct75,"75th pct","#4a9edd",1),
        (pct50,"Median","#ddeeff",2.5),
        (pct25,"25th pct","#f5a623",1),(pct10,"10th pct","#e05c3a",1.5),
    ]:
        fig_mc.add_trace(go.Scatter(x=x_yrs,y=pct_vals,name=pct_name,
            line=dict(color=col,width=width)))
    # Starting value line
    fig_mc.add_hline(y=mc_value,line_color="#4a6fa5",line_dash="dot",line_width=1)

    fig_mc.update_layout(**PL(height=420,hovermode="x unified",
        title=dict(text=f"🎲 {mc_n:,} Monte Carlo Paths  ·  {mc_years}yr horizon  ·  "
                   f"${mc_contrib:,}/mo contribution",font=dict(color="#ddeeff",size=13))))
    fig_mc.update_xaxes(title="Years",**GRID)
    fig_mc.update_yaxes(title="Portfolio Value ($)",tickprefix="$",tickformat=",.0f",**GRID)
    st.plotly_chart(fig_mc,width='stretch')

    # Summary stats
    final=all_paths[:,-1]
    prob_double = (final >= mc_value*2).mean()*100
    prob_lose20 = (final <= mc_value*0.8).mean()*100
    prob_pos    = (final >= mc_value).mean()*100

    r1,r2,r3,r4,r5,r6=st.columns(6)
    r1.metric("Median Final",   f"${np.median(final):,.0f}")
    r2.metric("90th Pct",       f"${np.percentile(final,90):,.0f}")
    r3.metric("10th Pct",       f"${np.percentile(final,10):,.0f}")
    r4.metric("Prob Positive",  f"{prob_pos:.0f}%",  delta_color="off")
    r5.metric("Prob 2× Money",  f"{prob_double:.0f}%",delta_color="off")
    r6.metric("Prob −20%+",     f"{prob_lose20:.0f}%",delta_color="off")

    # Histogram of final values
    fig_hist=go.Figure(go.Histogram(x=final,nbinsx=60,marker_color="#4a9edd",opacity=0.8,
        marker_line_width=0))
    fig_hist.add_vline(x=mc_value,line_color="#f5d76e",line_dash="dash",
        annotation_text="Start",annotation_font_color="#f5d76e",annotation_font_size=9)
    fig_hist.add_vline(x=np.median(final),line_color="#10b981",line_dash="dash",
        annotation_text="Median",annotation_font_color="#10b981",annotation_font_size=9)
    fig_hist.update_layout(**PL(height=220,
        title=dict(text=f"Distribution of Final Values at Year {mc_years}",
                   font=dict(color="#ddeeff",size=11))))
    fig_hist.update_xaxes(title="Final Value ($)",tickprefix="$",tickformat=",.0f",**GRID)
    fig_hist.update_yaxes(title="Simulations",**GRID)
    st.plotly_chart(fig_hist,width='stretch')

# ══ TAB 8: FACTOR REGRESSION ═════════════════════════════════════════════════
with tab8:
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px'>🧬 FACTOR REGRESSION</p>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-bottom:10px'>"
                "Decomposes your portfolio returns into underlying risk factors using ETF proxies from your own price data. "
                "<b style='color:#8da0bc'>MKT</b> = market beta (SPY excess return). "
                "<b style='color:#8da0bc'>SMB</b> = small-cap tilt (AVUV minus SPY). "
                "<b style='color:#8da0bc'>MOM</b> = momentum tilt (QMOM minus SPY). "
                "<b style='color:#8da0bc'>DUR</b> = duration/bond factor (IEF). "
                "<b style='color:#8da0bc'>Alpha</b> = return unexplained by any factor — the holy grail."
                "</div>",unsafe_allow_html=True)

    # Build factor proxies from local ETF data — no internet required
    ff = None
    needed = ["SPY","AVUV","QMOM","IEF"]
    if all(t in prices.columns for t in needed):
        cols = list(dict.fromkeys([t for t in TICKERS_LIST if t in prices.columns]+["SPY"]))
        px_f = prices[cols].copy()
        px_f = px_f[~px_f.index.duplicated(keep="last")]
        px_f = px_f.pct_change(fill_method=None).dropna()
        rf_daily = CASH_RATE_REAL / 100 / 252
        spy_s = px_f["SPY"].copy()
        ff = pd.DataFrame({
            "Mkt-RF": spy_s - rf_daily,
            "SMB":    px_f["AVUV"].values - spy_s.values,
            "MOM":    px_f["QMOM"].values - spy_s.values,
            "DUR":    px_f["IEF"].values,
        }, index=px_f.index).dropna()

    if ff is None or ff.empty:
        st.warning("Need SPY, AVUV, QMOM and IEF in price data. Run `python3 -m data.fetcher`.")
    else:
        # Align portfolio returns with FF factors
        pr_ff=port_ret.copy()
        common=pr_ff.index.intersection(ff.index)
        if len(common)<60:
            st.warning("Not enough overlapping data for regression (need >60 days).")
        else:
            pr_ff=pr_ff.loc[common]
            ff_c=ff.loc[common]
            # Local proxies: Mkt-RF, SMB, MOM, DUR — no RF column needed
            y=pr_ff.values  # raw portfolio return (rf already baked into Mkt-RF proxy)
            FACTOR_COLS=["Mkt-RF","SMB","MOM","DUR"]
            X=ff_c[FACTOR_COLS].values
            X_const=np.column_stack([np.ones(len(X)),X])
            try:
                coeffs=np.linalg.lstsq(X_const,y,rcond=None)[0]
                y_hat=X_const@coeffs
                resid=y-y_hat
                ss_res=np.sum(resid**2); ss_tot=np.sum((y-y.mean())**2)
                r2=1-ss_res/ss_tot if ss_tot>0 else 0
                n_obs=len(y); n_params=len(coeffs)
                try:
                    vcv=np.linalg.inv(X_const.T@X_const)*(ss_res/(n_obs-n_params))
                    se=np.sqrt(np.diag(vcv))
                except np.linalg.LinAlgError:
                    se=np.ones(n_params)*np.nan
                t_stats=coeffs/se
                from scipy import stats as scipy_stats
                p_vals=[float(2*(1-scipy_stats.t.cdf(abs(t),df=max(n_obs-n_params,1)))) for t in t_stats]

                factor_names=["Alpha (ann.)","Mkt-RF (Beta)","SMB (Size)","MOM (Momentum)","DUR (Duration)"]
                coeff_display=[coeffs[0]*252]+list(coeffs[1:])
                se_display   =[se[0]*252]+list(se[1:])

                fa,fb,fc=st.columns(3)
                fa.metric("R² (factor explained)",f"{r2:.1%}")
                fb.metric("Idiosyncratic (1-R²)",f"{1-r2:.1%}")
                fc.metric("Ann. Alpha",f"{coeff_display[0]*100:+.2f}%",
                          "***" if p_vals[0]<0.01 else "n.s.",delta_color="off")

                # Factor exposure bar
                exposures=list(coeffs[1:])
                t_vals=list(t_stats[1:])
                fig_f=go.Figure()
                fig_f.add_trace(go.Bar(x=factor_names[1:],y=exposures,
                    marker_color=["#10b981" if v>0 else "#e05c3a" for v in exposures],
                    marker_line_width=0,
                    text=[f"{v:+.3f} (t={t:.1f})" for v,t in zip(exposures,t_vals)],
                    textposition="outside",textfont=dict(color="#8da0bc",size=10)))
                fig_f.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
                fig_f.add_hline(y=1,line_color="#4a6fa5",line_dash="dash",line_width=0.5,
                    annotation_text="beta=1",annotation_font_color="#4a6fa5",annotation_font_size=9)
                fig_f.update_layout(**PL(height=300,
                    title=dict(text=f"Factor Loadings  ·  R²={r2:.1%}  ·  Alpha={coeff_display[0]*100:+.2f}%/yr",
                               font=dict(color="#ddeeff",size=12))))
                fig_f.update_xaxes(**GRID)
                fig_f.update_yaxes(title="Loading",**GRID)
                st.plotly_chart(fig_f,width='stretch')

                # Regression table
                reg_rows=[]
                for name,coef,s_e,t_s,p_v in zip(factor_names,coeff_display,se_display,t_stats,p_vals):
                    sig="***" if p_v<0.01 else "**" if p_v<0.05 else "*" if p_v<0.10 else ""
                    reg_rows.append({"Factor":name,"Coeff":f"{coef:+.4f}",
                        "Std Err":f"{s_e:.4f}","t-stat":f"{t_s:+.2f}",
                        "p-val":f"{p_v:.3f}","Sig":sig})
                st.dataframe(pd.DataFrame(reg_rows).set_index("Factor"),width='stretch',height=220)

                st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-top:6px'>"
                    "* p&lt;0.10  ** p&lt;0.05  *** p&lt;0.01  ·  "
                    "Mkt-RF = SPY excess return proxy  ·  SMB = AVUV minus SPY  ·  "
                    "MOM = QMOM minus SPY  ·  DUR = IEF  ·  Alpha annualised × 252"
                    "</div>",unsafe_allow_html=True)

                # Rolling factor exposures
                st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:12px'>"
                    "ROLLING 252-DAY FACTOR EXPOSURES</p>",unsafe_allow_html=True)
                roll_window=min(252,len(y)//2)
                roll_betas={fn:[] for fn in FACTOR_COLS}
                roll_dates=[]
                ff_arr=ff_c[FACTOR_COLS].values
                for i in range(roll_window,len(y)+1):
                    y_w=y[i-roll_window:i]; X_w=ff_arr[i-roll_window:i]
                    Xc=np.column_stack([np.ones(roll_window),X_w])
                    try:
                        b=np.linalg.lstsq(Xc,y_w,rcond=None)[0]
                        for j,fn in enumerate(FACTOR_COLS):
                            roll_betas[fn].append(b[j+1])
                        roll_dates.append(pr_ff.index[i-1])
                    except: pass

                if roll_dates:
                    fig_rb=go.Figure()
                    roll_cols_map={"Mkt-RF":"#4a9edd","SMB":"#10b981","MOM":"#a855f7","DUR":"#f5d76e"}
                    for fn,col in roll_cols_map.items():
                        if roll_betas[fn]:
                            fig_rb.add_trace(go.Scatter(x=roll_dates,y=roll_betas[fn],
                                name=fn,line=dict(color=col,width=1.5)))
                    fig_rb.add_hline(y=0,line_color="#4a6fa5",line_dash="dot")
                    fig_rb.update_layout(**PL(height=280,hovermode="x unified",
                        title=dict(text="Rolling 252-day Factor Loadings",font=dict(color="#ddeeff",size=11))))
                    fig_rb.update_xaxes(**GRID); fig_rb.update_yaxes(**GRID)
                    st.plotly_chart(fig_rb,width='stretch')
            except Exception as e:
                st.error(f"Regression failed: {e}")

# ══ TAB 9: REBALANCE ═════════════════════════════════════════════════════════
with tab9:
    hd=load_holdings(); initialized=hd.get("initialized",False)
    actual_w=pd.Series(hd.get("weights",{})).reindex(TICKERS_LIST).fillna(0)
    if not initialized:
        st.markdown("""<div style='background:#0d1525;border:1px solid #f5d76e;border-left:3px solid #f5d76e;
            border-radius:6px;padding:16px;margin-bottom:16px'>
            <div style='font-size:13px;font-weight:700;color:#f5d76e;margin-bottom:6px'>🚀 First Time Setup</div>
            <div style='font-size:12px;color:#8da0bc'>Set actual holdings = target weights as day-1 baseline.</div>
        </div>""",unsafe_allow_html=True)
        if st.button("🚀 Initialize to Target Weights",width='stretch'):
            initialize_to_target(target_w.to_dict()); st.success("✅ Done!"); st.rerun()
    else:
        st.caption(f"Last updated: {hd.get('last_updated','unknown')}")

    if initialized:
        portfolio_value=st.number_input("Portfolio Value ($)",value=100_000,step=10_000,format="%d")

        st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:8px'>📋 HOLDINGS</p>",
                    unsafe_allow_html=True)
        hrows=[]
        for t in TICKERS_LIST:
            tw=float(target_w.get(t,0)); aw=float(actual_w.get(t,0))
            tval=tw*portfolio_value; aval=aw*portfolio_value
            lpx=float(prices[t].iloc[-1]) if t in prices.columns else None
            hrows.append({
                "Ticker":t,"Name":TICKERS[t]["label"],
                "Last Px":  f"${lpx:.2f}" if lpx else "—",
                "Target %": f"{tw:.1%}","Actual %":f"{aw:.1%}","Diff":f"{tw-aw:+.1%}",
                "Tgt Shares":round(tval/lpx,2) if lpx and lpx>0 else "—",
                "Act Shares":round(aval/lpx,2) if lpx and lpx>0 else "—",
                "Target $":  f"${tval:,.0f}","Actual $":f"${aval:,.0f}",
                "Delta $":   f"${tval-aval:+,.0f}",
            })
        st.dataframe(pd.DataFrame(hrows).set_index("Ticker"),width='stretch',height=320)

        drift=(target_w-actual_w)*100
        fig_d=go.Figure(go.Bar(x=drift.reindex(TICKERS_LIST).index,y=drift.reindex(TICKERS_LIST).values,
            marker_color=["#10b981" if v>0 else "#e05c3a" if v<0 else "#4a6fa5"
                          for v in drift.reindex(TICKERS_LIST).values],
            marker_line_width=0,text=[f"{v:+.1f}%" for v in drift.reindex(TICKERS_LIST).values],
            textposition="outside",textfont=dict(color="#8da0bc",size=10)))
        fig_d.add_hline(y=2, line_color="#f5d76e",line_dash="dash",line_width=1)
        fig_d.add_hline(y=-2,line_color="#f5d76e",line_dash="dash",line_width=1)
        fig_d.add_hline(y=0, line_color="#4a6fa5",line_dash="dot")
        fig_d.update_layout(**PL(height=260,title=dict(text="⚖️ Weight Drift (±2% threshold)",
            font=dict(color="#ddeeff",size=12))))
        fig_d.update_xaxes(**GRID); fig_d.update_yaxes(ticksuffix="%",**GRID)
        st.plotly_chart(fig_d,width='stretch')

        trades=generate_trades(target_w,actual_w,portfolio_value)
        needs_reb=trades[trades["trade_required"]]
        if needs_reb.empty: st.success("✅ All within threshold.")
        else: st.warning(f"⚠️ {len(needs_reb)} position(s) beyond ±2%")
        st.dataframe(trades[["action","current_weight","target_weight","drift","trade_value"]]
            .style.format({"current_weight":"{:.1%}","target_weight":"{:.1%}",
                           "drift":"{:+.1%}","trade_value":"${:,.0f}"}),
            width='stretch',height=260)

        st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:12px'>✏️ LOG A TRADE</p>",
                    unsafe_allow_html=True)
        tc1,tc2,tc3,tc4=st.columns(4)
        with tc1: trade_t=st.selectbox("Ticker",TICKERS_LIST)
        with tc2: new_pct=st.number_input("New Weight (%)",value=float(round(actual_w.get(trade_t,0)*100,1)),
                                           min_value=0.0,max_value=200.0,step=0.1)
        with tc3: note=st.text_input("Note",placeholder="rebalanced / new position")
        with tc4:
            st.markdown("<div style='height:28px'></div>",unsafe_allow_html=True)
            if st.button("💾 Save",width='stretch'):
                nw=actual_w.to_dict(); nw[trade_t]=new_pct/100
                update_weights(nw,note=note or f"Updated {trade_t}")
                st.success(f"✅ {trade_t} → {new_pct:.1f}%"); st.rerun()

        log=get_trade_log()
        if log:
            st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:8px'>📋 TRADE LOG</p>",
                        unsafe_allow_html=True)
            ldf=pd.DataFrame(reversed(log[-20:]))
            ldf["date"]=pd.to_datetime(ldf["date"]).dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(ldf[[c for c in ["date","action","ticker","old_weight","new_weight","note"]
                               if c in ldf.columns]],width='stretch',height=200)

# ══ TAB 10: FRONTIER + SML ═══════════════════════════════════════════════════
with tab10:
    n_t=len(TICKERS_LIST)
    va=np.array([VOLS[t] for t in TICKERS_LIST])
    ra=np.array([EXPECTED_RETURNS[t] for t in TICKERS_LIST])
    cov_f=np.outer(va,va)*CORR_MATRIX.values/10000
    np.random.seed(42)
    sv,sr_,ss=[],[],[]
    for _ in range(2000):
        w=np.random.dirichlet(np.ones(n_t))
        pv=np.sqrt(max(w@cov_f@w,1e-10))*100; pr=w@ra
        sv.append(pv); sr_.append(pr); ss.append((pr-CASH_RATE_REAL)/pv)

    bench_pts={}
    for bname in BENCHMARKS:
        if bname in prices.columns:
            br=prices[bname].pct_change(fill_method=None).dropna()
            bench_pts[bname]=(br.std()*np.sqrt(252)*100, br.mean()*252*100)

    fig1=go.Figure()
    fig1.add_trace(go.Scatter(x=sv,y=sr_,mode="markers",
        marker=dict(size=4,color=ss,colorscale="Viridis",opacity=0.5,
            colorbar=dict(title=dict(text="Sharpe",font=dict(color="#8da0bc")),
                tickfont=dict(color="#8da0bc"),x=1.02,thickness=14),showscale=True),
        name="Simulated",hovertemplate="Vol:%{x:.1f}%  Ret:%{y:.1f}%<extra></extra>"))
    for t in TICKERS_LIST:
        fig1.add_trace(go.Scatter(x=[VOLS[t]],y=[EXPECTED_RETURNS[t]],mode="markers+text",
            marker=dict(size=10,color=TICKER_COL[t],line=dict(color="#080c14",width=1.5)),
            text=[t],textposition="top center",textfont=dict(color="#ddeeff",size=9),
            showlegend=False,hovertemplate=f"<b>{t}</b><br>Vol:{VOLS[t]}%  Ret:{EXPECTED_RETURNS[t]}%<extra></extra>"))
    for bname,(bvol,bret) in bench_pts.items():
        fig1.add_trace(go.Scatter(x=[bvol],y=[bret],mode="markers+text",
            marker=dict(size=13,color=BENCHMARKS[bname],symbol="diamond",line=dict(color="#080c14",width=2)),
            text=[bname],textposition="top center",textfont=dict(color=BENCHMARKS[bname],size=9),
            name=bname,showlegend=True))
    pv=d["stats"]["volatility"]; pr_s=d["stats"]["expected_return"]
    fig1.add_trace(go.Scatter(x=[pv],y=[pr_s],mode="markers",
        marker=dict(size=18,color="#ffffff",symbol="star",line=dict(color="#4a9edd",width=2)),
        name="★ Max Sharpe"))
    cx=np.linspace(0,30,100)
    fig1.add_trace(go.Scatter(x=cx,y=CASH_RATE_REAL+d["stats"]["sharpe_ratio"]*cx,
        mode="lines",line=dict(color="#4a9edd",width=1.5,dash="dash"),name="CML",hoverinfo="skip"))
    fig1.update_layout(**PL(height=480,margin=dict(l=60,r=90,t=50,b=60),
        title=dict(text="🗺️ Efficient Frontier  ·  2,000 Portfolios  ·  Colour = Sharpe",
                   font=dict(color="#ddeeff",size=13))))
    fig1.update_xaxes(title="Volatility (%)",ticksuffix="%",range=[3,28],**GRID)
    fig1.update_yaxes(title="Exp. Real Return (%)",ticksuffix="%",**GRID)
    st.plotly_chart(fig1,width='stretch')

    # Security Market Line
    st.markdown("<p style='font-size:10px;color:#4a6fa5;letter-spacing:2px;margin-top:8px'>"
                "📏 SECURITY MARKET LINE</p>",unsafe_allow_html=True)
    st.markdown("<div style='font-size:10px;color:#4a6fa5;margin-bottom:6px'>"
                "<b style='color:#8da0bc'>Beta</b> = sensitivity to SPY. "
                "Assets <b style='color:#10b981'>above the SML</b> offer more return than CAPM predicts "
                "(positive alpha). Assets below are overpaying for market risk.</div>",unsafe_allow_html=True)

    if "SPY" in prices.columns:
        spy_r_sml=prices["SPY"].pct_change(fill_method=None).dropna()
        mkt_prem=EXPECTED_RETURNS.get("SPY",3.5)-CASH_RATE_REAL
        betas,exp_rets_sml,labels,colours=[],[],[],[]
        for t in TICKERS_LIST:
            if t in prices.columns:
                tr=prices[t].pct_change(fill_method=None).dropna()
                idx=tr.index.intersection(spy_r_sml.index)
                if len(idx)>60:
                    cv=np.cov(tr.loc[idx],spy_r_sml.loc[idx])[0,1]
                    vr=np.var(spy_r_sml.loc[idx])
                    betas.append(cv/vr if vr>0 else 1.0)
                    exp_rets_sml.append(EXPECTED_RETURNS[t])
                    labels.append(t); colours.append(TICKER_COL[t])
        if betas:
            sml_x=np.linspace(0,max(betas)*1.15,100)
            fig2=go.Figure()
            fig2.add_trace(go.Scatter(x=sml_x,y=CASH_RATE_REAL+sml_x*mkt_prem,mode="lines",
                line=dict(color="#4a6fa5",width=1.5,dash="dash"),name="SML (CAPM)",hoverinfo="skip"))
            for i,t in enumerate(labels):
                capm_ret=CASH_RATE_REAL+betas[i]*mkt_prem
                alpha=exp_rets_sml[i]-capm_ret; above=alpha>=0
                fig2.add_trace(go.Scatter(x=[betas[i]],y=[exp_rets_sml[i]],mode="markers+text",
                    marker=dict(size=12,color=colours[i],
                                line=dict(color="#10b981" if above else "#e05c3a",width=2)),
                    text=[t],textposition="top center",textfont=dict(color="#ddeeff",size=9),
                    showlegend=False,
                    hovertemplate=f"<b>{t}</b><br>Beta:{betas[i]:.2f}  Exp:{exp_rets_sml[i]:.1f}%  "
                                  f"CAPM:{capm_ret:.1f}%  Alpha:{alpha:+.1f}%<extra></extra>"))
            for bname,(bvol_b,bret_b) in bench_pts.items():
                if bname in prices.columns:
                    br2=prices[bname].pct_change(fill_method=None).dropna()
                    idx2=br2.index.intersection(spy_r_sml.index)
                    if len(idx2)>60:
                        cv2=np.cov(br2.loc[idx2],spy_r_sml.loc[idx2])[0,1]
                        vr2=np.var(spy_r_sml.loc[idx2])
                        beta_b=cv2/vr2 if vr2>0 else 1.0
                        fig2.add_trace(go.Scatter(x=[beta_b],y=[bret_b],mode="markers+text",
                            marker=dict(size=13,color=BENCHMARKS[bname],symbol="diamond",
                                        line=dict(color="#080c14",width=2)),
                            text=[bname],textposition="top center",
                            textfont=dict(color=BENCHMARKS[bname],size=9),
                            name=bname,showlegend=True))
            fig2.update_layout(**PL(height=420,margin=dict(l=60,r=30,t=50,b=60),
                title=dict(text="📏 Security Market Line  ·  Green border = above CAPM",
                           font=dict(color="#ddeeff",size=12))))
            fig2.update_xaxes(title="Beta (vs SPY)",**GRID)
            fig2.update_yaxes(title="Exp. Real Return (%)",ticksuffix="%",**GRID)
            st.plotly_chart(fig2,width='stretch')

st.markdown("<div style='border-top:1px solid #162030;margin-top:24px;padding-top:10px;"
            "font-size:10px;color:#2a3f5a'>AQR Alternative Thinking 2026  ·  "
            "Not investment advice  ·  For illustrative purposes only</div>",unsafe_allow_html=True)
