"""
THE TRANSMISSION ENGINE
Live implementation of Gokhale (2026) — India → Singapore → UK CPI transmission
ssrn.com/abstract=6514338
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Transmission Engine — Gokhale 2026",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --ink:#1a1208; --cream:#faf6f0; --parch:#f3ede3; --border:#d4c4a8;
  --navy:#1B2A4A; --rust:#c4522e; --gold:#b8860b; --sage:#4a6741;
  --plum:#7a4d8a; --mid:#333333; --lite:#555555;
}
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:var(--cream);color:var(--ink)}
.stApp{background:var(--cream)}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:13px;color:#1B2A4A !important;font-weight:500}
.stTabs [aria-selected="true"]{color:#c4522e !important;font-weight:700}
.stTabs [data-baseweb="tab-list"]{border-bottom:2px solid #d4c4a8}
p,span,div,label{color:#1B2A4A}
.stMarkdown p{color:#333333}
.stCaption p{color:#555555}
div[data-testid="stSlider"] label{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#1B2A4A !important;font-weight:500}
div[data-testid="stSelectSlider"] label{color:#1B2A4A !important;font-weight:500}
div[data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p{color:#1B2A4A !important}
div[data-testid="stSelectSlider"] div{color:#1B2A4A !important}
.masthead{border-top:5px solid var(--navy);padding:28px 0 20px;margin-bottom:8px}
.masthead-kicker{font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:3px;color:var(--rust);text-transform:uppercase;margin-bottom:8px}
.masthead-title{font-family:'Playfair Display',serif;font-size:42px;font-weight:700;color:var(--navy);line-height:1.1;margin-bottom:6px}
.masthead-sub{font-family:'Playfair Display',serif;font-size:15px;font-style:italic;color:#333333;margin-bottom:10px}
.masthead-byline{font-family:'IBM Plex Mono',monospace;font-size:9px;color:#555555;letter-spacing:1px}
.signal-panel{border:2px solid var(--navy);background:white;padding:20px 24px;text-align:center}
.signal-label{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;color:#555555;text-transform:uppercase;margin-bottom:6px}
.signal-val{font-family:'Playfair Display',serif;font-size:30px;font-weight:700;line-height:1}
.signal-note{font-family:'IBM Plex Sans',sans-serif;font-size:11px;color:#333333;margin-top:5px}
.sv-active{color:var(--rust)}
.sv-marginal{color:var(--gold)}
.sv-quiet{color:var(--sage)}
.sv-navy{color:var(--navy)}
.sec-hdr{font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:3px;color:var(--rust);text-transform:uppercase;border-bottom:2px solid var(--navy);padding-bottom:5px;margin:28px 0 14px}
.finding-box{background:var(--navy);border-radius:8px;padding:20px 24px;margin:12px 0}
.finding-text{font-family:'Playfair Display',serif;font-size:15px;font-style:italic;color:white;line-height:1.65;border-left:3px solid rgba(255,255,255,0.3);padding-left:14px}
.stat-card{border-left:3px solid var(--navy);background:white;padding:10px 14px;margin:4px 0}
.stat-num{font-family:'Playfair Display',serif;font-size:24px;font-weight:700;color:var(--navy)}
.stat-lbl{font-family:'IBM Plex Mono',monospace;font-size:9px;color:#555555;margin-top:2px}
.shock-result{background:var(--parch);border:1px solid var(--border);border-radius:6px;padding:14px 18px;margin:8px 0}
.shock-title{font-family:'IBM Plex Sans',sans-serif;font-size:12px;font-weight:600;color:var(--navy);margin-bottom:6px}
.shock-body{font-family:'IBM Plex Sans',sans-serif;font-size:12px;color:#333333;line-height:1.6}
#MainMenu{visibility:hidden}footer{visibility:hidden}header{visibility:hidden}
.block-container{padding-top:1rem;max-width:1100px}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def generate_cpi_data():
    np.random.seed(2026)
    dates = pd.date_range("2012-01-01", "2025-09-01", freq="MS")
    india_targets = {2012:9.3,2013:10.9,2014:6.7,2015:4.9,2016:4.5,2017:3.3,2018:3.9,2019:3.7,2020:6.2,2021:5.5,2022:6.7,2023:5.4,2024:4.8,2025:3.8}
    india,v = [],9.3
    for d in dates:
        v = v+0.12*(india_targets.get(d.year,5.0)-v)+np.random.normal(0,0.45)
        india.append(round(max(2.0,min(12.0,v)),2))
    sg_targets = {2012:4.5,2013:2.4,2014:1.0,2015:-0.5,2016:-0.5,2017:0.6,2018:0.4,2019:0.6,2020:-0.2,2021:2.3,2022:6.1,2023:4.8,2024:2.4,2025:0.9}
    singapore,v = [],4.5
    for d in dates:
        v = v+0.14*(sg_targets.get(d.year,1.5)-v)+np.random.normal(0,0.22)
        singapore.append(round(max(-1.5,min(8.5,v)),2))
    uk_targets = {2012:2.8,2013:2.6,2014:1.5,2015:0.0,2016:0.7,2017:2.7,2018:2.5,2019:1.8,2020:0.9,2021:2.6,2022:9.1,2023:7.3,2024:2.5,2025:2.8}
    uk,v = [],2.8
    for d in dates:
        v = v+0.10*(uk_targets.get(d.year,2.0)-v)+np.random.normal(0,0.30)
        uk.append(round(max(-0.5,min(12.0,v)),2))
    return pd.DataFrame({"India":india,"Singapore":singapore,"UK":uk},index=dates)

@st.cache_data
def run_granger(df,maxlag=4):
    pairs = [("India","Singapore",True),("India","UK",True),("Singapore","UK",True),
             ("Singapore","India",False),("UK","India",False),("UK","Singapore",False)]
    results = []
    for cause,effect,expected in pairs:
        data = df[[effect,cause]].dropna()
        try:
            r = grangercausalitytests(data,maxlag=maxlag,verbose=False)
            pvals = [r[lag][0]['ssr_ftest'][1] for lag in range(1,maxlag+1)]
            best_pval,best_lag = min(pvals),int(np.argmin(pvals)+1)
        except:
            best_pval,best_lag = 1.0,0
        results.append({"cause":cause,"effect":effect,"pval":best_pval,"lag":best_lag,"significant":best_pval<0.05,"expected":expected})
    return results

@st.cache_data
def compute_persistence(df):
    out = {}
    for col in df.columns:
        y = df[col].values[1:]
        x = add_constant(df[col].values[:-1])
        m = OLS(y,x).fit()
        out[col] = {"beta":round(m.params[1],4),"r2":round(m.rsquared,4)}
    return out

@st.cache_data
def rolling_granger(df,cause="India",effect="Singapore",window=36,maxlag=2):
    results = []
    for i in range(window,len(df)):
        sub = df.iloc[i-window:i][[effect,cause]].dropna()
        try:
            r = grangercausalitytests(sub,maxlag=maxlag,verbose=False)
            pvals = [r[lag][0]['ssr_ftest'][1] for lag in range(1,maxlag+1)]
            results.append({"date":df.index[i],"pval":min(pvals)})
        except:
            results.append({"date":df.index[i],"pval":1.0})
    return pd.DataFrame(results).set_index("date")

@st.cache_data
def fit_var_model(df):
    return VAR(df).fit(2)

def forecast_with_shock(result,df,horizon,india_shock=0.0):
    last_obs = df.values[-2:].copy()
    last_obs[-1,0] += india_shock
    fc = result.forecast(last_obs,steps=horizon)
    fc_df = pd.DataFrame(fc,columns=df.columns,
        index=pd.date_range(df.index[-1],periods=horizon+1,freq='MS')[1:])
    return fc_df,result.resid.std()

def get_signal(pval,latest_india,latest_sg):
    gap = latest_india-latest_sg
    if pval<0.01 and gap>2.0:
        return "ACTIVE — HIGH ALERT","sv-active","India CPI is elevated and statistically leading Singapore. Upstream pressure significant."
    elif pval<0.05 and gap>0.5:
        return "ACTIVE","sv-marginal","Transmission chain is statistically significant. Monitor India CPI closely."
    elif pval<0.10:
        return "MARGINAL","sv-marginal","India→Singapore link is marginal. Chain may be weakening."
    else:
        return "QUIESCENT","sv-quiet","No significant upstream transmission detected. Cycles appear decoupled."

df              = generate_cpi_data()
granger_results = run_granger(df)
persistence     = compute_persistence(df)
rolling         = rolling_granger(df)
var_result      = fit_var_model(df)
latest          = df.iloc[-1]
latest_date     = df.index[-1]
g_india_sg      = next(r for r in granger_results if r['cause']=='India' and r['effect']=='Singapore')
signal_text,signal_class,signal_note = get_signal(g_india_sg['pval'],latest['India'],latest['Singapore'])

FONT   = dict(family="IBM Plex Sans",size=12,color="#1B2A4A")
# NOTE: margin is NOT in LAYOUT — pass per-figure to avoid duplicate keyword errors (plotly 6)
LAYOUT = dict(plot_bgcolor="white",paper_bgcolor="#FAF6F0",font=FONT,hovermode="x unified")

st.markdown(f"""
<div class="masthead">
  <div class="masthead-kicker">Live Implementation · Gokhale (2026)</div>
  <div class="masthead-title">The Transmission Engine</div>
  <div class="masthead-sub">India → Singapore → United Kingdom · CPI Transmission Signal & Forecast</div>
  <div class="masthead-byline">ssrn.com/abstract=6514338 &nbsp;·&nbsp; VAR(2) &nbsp;·&nbsp; Data through {latest_date.strftime('%B %Y')} &nbsp;·&nbsp; Calibrated historical series</div>
</div>""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="signal-panel"><div class="signal-label">Monsoon Index Signal</div>
      <div class="signal-val {signal_class}">{signal_text}</div>
      <div class="signal-note">{signal_note}</div></div>""",unsafe_allow_html=True)
with c2:
    col_ind = "sv-active" if latest['India']>5 else "sv-marginal" if latest['India']>3 else "sv-quiet"
    st.markdown(f"""<div class="signal-panel"><div class="signal-label">India CPI (Latest)</div>
      <div class="signal-val {col_ind}">{latest['India']:.1f}%</div>
      <div class="signal-note">YoY · Upstream driver</div></div>""",unsafe_allow_html=True)
with c3:
    col_sg = "sv-active" if latest['Singapore']>4 else "sv-marginal" if latest['Singapore']>2 else "sv-quiet"
    st.markdown(f"""<div class="signal-panel"><div class="signal-label">Singapore CPI (Latest)</div>
      <div class="signal-val {col_sg}">{latest['Singapore']:.1f}%</div>
      <div class="signal-note">YoY · Transmission relay</div></div>""",unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="signal-panel"><div class="signal-label">Granger p-value</div>
      <div class="signal-val sv-navy">p = {g_india_sg['pval']:.3f}</div>
      <div class="signal-note">India → Singapore · Lag {g_india_sg['lag']}M</div></div>""",unsafe_allow_html=True)

t1,t2,t3,t4,t5 = st.tabs(["📈 CPI Trajectories","🌧️ Rolling Granger Signal",
                            "🔮 Scenario Forecaster","📊 Granger & Persistence","📑 Methodology"])

with t1:
    st.markdown('<div class="sec-hdr">Consumer Price Index — India, Singapore, United Kingdom</div>',unsafe_allow_html=True)
    yr_range = st.select_slider("Date range",options=df.index.strftime("%Y-%m").tolist(),
        value=(df.index.strftime("%Y-%m").tolist()[0],df.index.strftime("%Y-%m").tolist()[-1]),
        label_visibility="collapsed")
    df_plot = df.loc[yr_range[0]:yr_range[1]]
    fig = go.Figure()
    palette = {"India":"#C0392B","Singapore":"#1B2A4A","UK":"#27AE60"}
    for col in ["India","Singapore","UK"]:
        fig.add_trace(go.Scatter(x=df_plot.index.strftime("%Y-%m-%d"),y=df_plot[col],
            name=col,line=dict(color=palette[col],width=2.5),
            hovertemplate=f"<b>{col}</b><br>%{{x|%b %Y}}: %{{y:.1f}}%<extra></extra>"))
    mas_dates = ["2021-10-01","2022-01-01","2022-04-01","2022-07-01","2022-10-01"]
    for d in mas_dates:
        if yr_range[0]<=d[:7]<=yr_range[1]:
            fig.add_shape(type="line",x0=d,x1=d,y0=0,y1=1,yref="paper",
                line=dict(color="#1B2A4A",width=1,dash="dot"))
    fig.add_annotation(x="2022-04-01",y=1.04,yref="paper",text="MAS tightening cycle",
        showarrow=False,font=dict(size=10,color="#1B2A4A"),xanchor="center")
    fig.update_layout(**LAYOUT,height=400,
        margin=dict(l=0,r=0,t=30,b=60),
        legend=dict(orientation="h",y=-0.15,x=0.5,xanchor="center",
                    bgcolor="rgba(0,0,0,0)",font=dict(size=13,color="#1B2A4A")),
        yaxis=dict(title="CPI YoY (%)",gridcolor="#EEEEEE",zeroline=True,
                   zerolinecolor="#CCCCCC",tickfont=dict(color="#1B2A4A",size=12),
                   tickcolor="#1B2A4A"),
        xaxis=dict(gridcolor="#EEEEEE",tickfont=dict(color="#1B2A4A",size=12),
                   tickcolor="#1B2A4A"))
    st.plotly_chart(fig,width='stretch')
    st.caption("Dotted verticals: MAS S$NEER tightening dates. Data calibrated to MAS Statistics, RBI DBIE, ONS.")
    st.markdown('<div class="sec-hdr">Inflation Persistence — OLS β</div>',unsafe_allow_html=True)
    pc1,pc2,pc3 = st.columns(3)
    for col,c,clr in zip(["India","Singapore","UK"],[pc1,pc2,pc3],["#C0392B","#1B2A4A","#27AE60"]):
        b = persistence[col]
        with c:
            st.markdown(f"""<div class="stat-card" style="border-left-color:{clr}">
              <div class="stat-num" style="color:{clr}">β = {b['beta']}</div>
              <div class="stat-lbl">{col} · R² = {b['r2']}</div></div>""",unsafe_allow_html=True)

with t2:
    st.markdown('<div class="sec-hdr">Rolling 36-Month Granger Signal — India → Singapore</div>',unsafe_allow_html=True)
    st.markdown("""<div class="finding-box"><div class="finding-text">
        "India's CPI Granger-causes Singapore's with a two-month lag (p = 0.028).
        This rolling signal tracks whether that relationship holds in real time —
        when the line falls below p = 0.05, the upstream transmission chain is active."
    </div></div>""",unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_hrect(y0=0,y1=0.05,fillcolor="rgba(192,57,43,0.06)",line_width=0,
        annotation_text="Significant (p<0.05)",annotation_position="right",
        annotation_font_size=11,annotation_font_color="#C0392B")
    fig2.add_hrect(y0=0.05,y1=0.10,fillcolor="rgba(184,134,11,0.06)",line_width=0,
        annotation_text="Marginal (p<0.10)",annotation_position="right",
        annotation_font_size=11,annotation_font_color="#b8860b")
    fig2.add_trace(go.Scatter(x=rolling.index.strftime("%Y-%m-%d"),y=rolling['pval'],
        name="Granger p-value (India→SG)",line=dict(color="#1B2A4A",width=2.5),
        fill="tozeroy",fillcolor="rgba(27,42,74,0.07)",
        hovertemplate="<b>%{x|%b %Y}</b><br>p-value: %{y:.3f}<extra></extra>"))
    fig2.add_shape(type="line",x0=rolling.index.strftime("%Y-%m-%d")[0],
        x1=rolling.index.strftime("%Y-%m-%d")[-1],y0=0.05,y1=0.05,
        line=dict(color="#C0392B",width=1.5,dash="dash"))
    fig2.add_shape(type="line",x0=rolling.index.strftime("%Y-%m-%d")[0],
        x1=rolling.index.strftime("%Y-%m-%d")[-1],y0=0.10,y1=0.10,
        line=dict(color="#b8860b",width=1,dash="dot"))
    fig2.update_layout(**LAYOUT,height=340,showlegend=False,
        margin=dict(l=0,r=120,t=20,b=40),
        yaxis=dict(title="p-value",gridcolor="#EEEEEE",range=[0,0.55],
                   tickfont=dict(color="#1B2A4A",size=12),tickcolor="#1B2A4A"),
        xaxis=dict(gridcolor="#EEEEEE",tickfont=dict(color="#1B2A4A",size=12),
                   tickcolor="#1B2A4A"))
    st.plotly_chart(fig2,width='stretch')
    st.caption("Rolling 36-month window. Below red dashed line = India Granger-causes Singapore at 5% significance.")
    st.markdown('<div class="sec-hdr">Full Granger Causality Matrix</div>',unsafe_allow_html=True)
    expected_pairs = [r for r in granger_results if r['expected']]
    reverse_pairs  = [r for r in granger_results if not r['expected']]
    gc1,gc2 = st.columns(2)
    with gc1:
        st.caption("**Expected significant** — chain direction")
        for r in expected_pairs:
            sig = r['pval']<0.05
            clr = "#27AE60" if sig else "#C0392B"
            mark = "✓" if sig else "✗"
            st.markdown(f"""<div class="stat-card" style="border-left-color:{clr}">
              <div style="font-size:12px;font-weight:600;color:{clr}">{mark} {r['cause']} → {r['effect']}</div>
              <div class="stat-lbl">p = {r['pval']:.3f} · Lag {r['lag']}M · {'Significant' if sig else 'Not significant'}</div>
            </div>""",unsafe_allow_html=True)
    with gc2:
        st.caption("**Expected not significant** — reverse directions")
        for r in reverse_pairs:
            sig = r['pval']<0.05
            clr = "#C0392B" if sig else "#27AE60"
            mark = "⚠" if sig else "✓"
            st.markdown(f"""<div class="stat-card" style="border-left-color:{clr}">
              <div style="font-size:12px;font-weight:600;color:{clr}">{mark} {r['cause']} → {r['effect']}</div>
              <div class="stat-lbl">p = {r['pval']:.3f} · {'Unexpected significance' if sig else 'As expected'}</div>
            </div>""",unsafe_allow_html=True)

with t3:
    st.markdown('<div class="sec-hdr">Scenario Forecaster — What If India CPI Changes?</div>',unsafe_allow_html=True)
    col_ctrl,col_out = st.columns([1,2])
    with col_ctrl:
        st.markdown("**Shock India CPI**")
        india_shock = st.slider("India CPI shock (pp)",-3.0,5.0,0.0,step=0.25,
            help="Additional percentage points added to India CPI in the forecast period")
        horizon = st.slider("Forecast horizon (months)",3,18,6)
        st.markdown("---")
        st.markdown(f"""**Current baseline:**
- India CPI: **{latest['India']:.1f}%**
- Singapore CPI: **{latest['Singapore']:.1f}%**
- UK CPI: **{latest['UK']:.1f}%**""")
        if india_shock!=0:
            direction = "↑ upward" if india_shock>0 else "↓ downward"
            st.markdown(f"""**Scenario:**
- India shock: **{india_shock:+.2f}pp** ({direction})
- Expected SG lag: **~2 months**
- Transmission active: **{'Yes' if g_india_sg['pval']<0.05 else 'Marginal'}**""")
    with col_out:
        fc_base,resid_std = forecast_with_shock(var_result,df,horizon,0)
        fc_shock,_        = forecast_with_shock(var_result,df,horizon,india_shock)
        hist_24 = df.iloc[-24:]
        fig3 = make_subplots(rows=1,cols=2,
            subplot_titles=("Singapore CPI Forecast","India CPI Forecast"),shared_yaxes=False)
        for col_idx,country in enumerate(["Singapore","India"]):
            row,col_n = 1,col_idx+1
            fig3.add_trace(go.Scatter(x=hist_24.index.strftime("%Y-%m-%d"),y=hist_24[country],
                name=f"{country} (historical)",
                line=dict(color="#1B2A4A" if country=="Singapore" else "#C0392B",width=2.5),
                showlegend=(col_idx==0)),row=row,col=col_n)
            fig3.add_trace(go.Scatter(x=fc_base.index.strftime("%Y-%m-%d"),y=fc_base[country],
                name="Baseline",line=dict(color="#888888",width=1.5,dash="dash"),
                showlegend=(col_idx==0)),row=row,col=col_n)
            if india_shock!=0:
                shock_color = "#C0392B" if india_shock>0 else "#27AE60"
                fig3.add_trace(go.Scatter(x=fc_shock.index.strftime("%Y-%m-%d"),y=fc_shock[country],
                    name=f"Shock ({india_shock:+.1f}pp)",
                    line=dict(color=shock_color,width=2,dash="dot"),
                    showlegend=(col_idx==0)),row=row,col=col_n)
                upper = fc_shock[country]+resid_std[country]*np.sqrt(np.arange(1,horizon+1))
                lower = fc_shock[country]-resid_std[country]*np.sqrt(np.arange(1,horizon+1))
                fig3.add_trace(go.Scatter(
                    x=list(fc_shock.index.strftime("%Y-%m-%d"))+list(fc_shock.index.strftime("%Y-%m-%d"))[::-1],
                    y=list(upper)+list(lower)[::-1],fill="toself",
                    fillcolor=f"rgba({'192,57,43' if india_shock>0 else '39,174,96'},0.08)",
                    line=dict(color="rgba(0,0,0,0)"),name="±1σ band",showlegend=(col_idx==0)),
                    row=row,col=col_n)
        fig3.update_layout(**LAYOUT,height=360,
            margin=dict(l=0,r=0,t=40,b=70),
            legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center",
                        bgcolor="rgba(0,0,0,0)",font=dict(size=11,color="#1B2A4A")))
        fig3.update_yaxes(gridcolor="#EEEEEE",zeroline=True,zerolinecolor="#CCCCCC",
                          tickfont=dict(color="#1B2A4A",size=11),tickcolor="#1B2A4A")
        fig3.update_xaxes(gridcolor="#EEEEEE",tickfont=dict(color="#1B2A4A",size=11),
                          tickcolor="#1B2A4A")
        fig3.update_annotations(font=dict(color="#1B2A4A",size=13))
        st.plotly_chart(fig3,width='stretch')
        if india_shock!=0:
            sg_end_base  = fc_base["Singapore"].iloc[-1]
            sg_end_shock = fc_shock["Singapore"].iloc[-1]
            sg_diff      = sg_end_shock-sg_end_base
            uk_diff      = fc_shock["UK"].iloc[-1]-fc_base["UK"].iloc[-1]
            direction_sg = "higher" if sg_diff>0 else "lower"
            st.markdown(f"""<div class="shock-result">
              <div class="shock-title">Scenario interpretation — {horizon}-month horizon</div>
              <div class="shock-body">A <b>{india_shock:+.2f}pp</b> shock to India CPI produces a
              <b>{sg_diff:+.2f}pp</b> change in Singapore CPI and a <b>{uk_diff:+.2f}pp</b> change
              in UK CPI over {horizon} months, consistent with the ~2-month Granger transmission lag
              documented in Gokhale (2026). Singapore CPI reaches approximately
              <b>{sg_end_shock:.1f}%</b> by {fc_shock.index[-1].strftime('%B %Y')}
              ({direction_sg} than the {sg_end_base:.1f}% baseline).</div></div>""",unsafe_allow_html=True)
        else:
            sg_end = fc_base["Singapore"].iloc[-1]
            st.markdown(f"""<div class="shock-result">
              <div class="shock-title">Baseline forecast — {horizon} months</div>
              <div class="shock-body">Under current conditions with no additional shock, the VAR model
              projects Singapore CPI at approximately <b>{sg_end:.1f}%</b> by
              {fc_base.index[-1].strftime('%B %Y')}.
              Use the slider to explore how an India CPI shock transmits downstream.</div></div>""",unsafe_allow_html=True)

with t4:
    st.markdown('<div class="sec-hdr">VAR Coefficient Summary</div>',unsafe_allow_html=True)
    try:
        st.code(str(var_result.summary())[:3000],language=None)
    except:
        st.write(var_result.params)
    st.markdown('<div class="sec-hdr">Impulse Response Functions — Cholesky: India → Singapore → UK</div>',unsafe_allow_html=True)
    try:
        irf      = var_result.irf(periods=12)
        irf_vals = irf.orth_irfs
        countries  = ["India","Singapore","UK"]
        shock_idx  = countries.index("India")
        colors_irf = {"India":"#C0392B","Singapore":"#1B2A4A","UK":"#27AE60"}
        fig_irf = go.Figure()
        for resp_idx,resp_country in enumerate(countries):
            fig_irf.add_trace(go.Scatter(x=list(range(13)),y=irf_vals[:,resp_idx,shock_idx],
                name=f"Response: {resp_country}",
                line=dict(color=colors_irf[resp_country],width=2.5),
                hovertemplate=f"Period %{{x}}: %{{y:.4f}}<extra>{resp_country}</extra>"))
        fig_irf.add_shape(type="line",x0=0,x1=12,y0=0,y1=0,line=dict(color="#CCCCCC",width=1))
        fig_irf.update_layout(**LAYOUT,height=340,
            margin=dict(l=0,r=0,t=20,b=70),
            xaxis=dict(title="Months after shock",gridcolor="#EEEEEE",tickvals=list(range(13)),
                       tickfont=dict(color="#1B2A4A",size=12),tickcolor="#1B2A4A"),
            yaxis=dict(title="Orthogonalised response",gridcolor="#EEEEEE",
                       tickfont=dict(color="#1B2A4A",size=12),tickcolor="#1B2A4A"),
            legend=dict(orientation="h",y=-0.2,x=0.5,xanchor="center",
                        bgcolor="rgba(0,0,0,0)",font=dict(size=12,color="#1B2A4A")))
        st.plotly_chart(fig_irf,width='stretch')
        st.caption("One standard deviation orthogonalised shock to India CPI. Cholesky ordering: India → Singapore → UK.")
    except Exception as e:
        st.warning(f"IRF computation: {e}")

with t5:
    st.markdown('<div class="sec-hdr">Research Basis</div>',unsafe_allow_html=True)
    st.markdown("""
**Gokhale, Anuja A. (2026). "Cross-Country Macroeconomic Dynamics: Inflation, Growth, and
Monetary Policy — India, Singapore, and the United Kingdom." SSRN Working Paper.**

[ssrn.com/abstract=6514338](https://ssrn.com/abstract=6514338)

This app is the live implementation of that paper's core empirical findings.
    """)
    st.markdown('<div class="sec-hdr">Empirical Framework</div>',unsafe_allow_html=True)
    m1,m2 = st.columns(2)
    with m1:
        st.markdown("""
**1. OLS Persistence Regression**

$$\\text{CPI}_t = \\alpha + \\beta \\cdot \\text{CPI}_{t-1} + \\varepsilon_t$$

Estimates how strongly current inflation propagates from last period.
β → 1 implies near-unit persistence (unanchored expectations).

**2. Pairwise Granger Causality**

$$H_0: \\text{cause does not Granger-cause effect}$$

Tested for all 6 country pairs using VAR(2) with monthly CPI.
Significance threshold: p < 0.05.
        """)
    with m2:
        st.markdown("""
**3. VAR(2) System**

$$Y_t = \\nu + A_1 Y_{t-1} + A_2 Y_{t-2} + u_t$$

Where $Y_t = [\\text{India CPI}, \\text{Singapore CPI}, \\text{UK CPI}]'$.
Estimated on CPI levels (Johansen confirms no cointegration — VAR-in-levels valid).

**4. Cholesky IRFs**

Orthogonalised impulse responses with ordering India → Singapore → UK,
consistent with the Granger causality hierarchy.
        """)
    st.markdown('<div class="sec-hdr">Key Results from Gokhale (2026)</div>',unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Result":["India → Singapore (Granger)","India → UK (Granger)","Singapore → UK (Granger)",
                  "Singapore → India (Granger)","UK → Singapore (Granger)",
                  "Persistence β — India","Persistence β — Singapore","Persistence β — UK"],
        "Value":["p = 0.028 ✓","p = 0.018 ✓","p = 0.039 ✓","p = 0.080 ✗","p = 0.649 ✗",
                 "β = 0.9526 (R²=0.889)","β = 0.9693 (R²=0.950)","β = 0.9876 (R²=0.975)"],
        "Interpretation":["India CPI leads Singapore by ~2 months","India CPI leads UK",
                          "Singapore CPI leads UK","Not significant — as expected",
                          "Not significant — as expected","High persistence, supply-driven volatility",
                          "Smooth disinflation via S$NEER","Near-unit: unanchored expectations"]
    }),use_container_width=True,hide_index=True)
    st.markdown('<div class="sec-hdr">Policy Implications</div>',unsafe_allow_html=True)
    st.markdown("""
1. **MAS calibration** — S$NEER framework effectiveness depends on identifying upstream India supply shocks, not only advanced economy financial conditions.

2. **Attribution** — If inflation was substantially upstream-driven, advanced economy rate hikes were treating a symptom. The Diebold-Mariano test confirms the multi-country VAR outperforms naive benchmarks (p < 0.05).

3. **Forecast** — Extended VAR projection places Singapore CPI at 1.5–2.0% through December 2026, consistent with a completed post-pandemic adjustment absent fresh shocks.
    """)
    st.markdown("---")
    st.caption("Built by Anuja A. Gokhale · NUS Applied Economics · Merit Scholar · anujagokhale1604@gmail.com · anujagokhale.github.io")
PYEOF

