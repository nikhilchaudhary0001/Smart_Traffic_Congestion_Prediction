"""TRAFFICIQ — a Streamlit command center for next-hour road probe-count forecasts.

Runs in preview mode when the original saved_model artifacts are absent.
The model and the original 17-feature calculation are never retrained or changed.
"""
from __future__ import annotations

import json
import gzip
import html
from datetime import date, datetime, time, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="TRAFFICIQ | Traffic intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "saved_model"
MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"
FEATURES = ["probeCount", "probeCount_lag_1", "probeCount_lag_2", "probeCount_lag_3", "probeCount_lag_24", "probeCount_rolling_mean_3h", "probeCount_rolling_std_3h", "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "speedLimit", "frc", "distance"]
BENCHMARK = {"Linear Regression": {"MAE": 24.9526, "RMSE": 46.8399, "R2": 0.9475}, "Polynomial Regression": {"MAE": 22.5873, "RMSE": 43.0991, "R2": 0.9556}, "Random Forest": {"MAE": 19.7777, "RMSE": 39.1660, "R2": 0.9633}}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root {color-scheme: dark;}
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
.stApp {background:#080e18;color:#e8f0fb;}
[data-testid="stSidebar"] {background:#0c1421;border-right:1px solid #1e3045;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {color:#9fb0c4;}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.45rem;max-width:1500px;padding-bottom:4rem;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.035em!important;}
h1 {font-size:2.3rem!important;} h2 {font-size:1.55rem!important;} h3 {font-size:1.15rem!important;}
[data-testid="stMetric"] {background:#101d2d;border:1px solid #203249;border-radius:14px;padding:17px 18px;min-height:122px;}
[data-testid="stMetricLabel"] {color:#8fa6bd!important;font-size:.83rem!important;}
[data-testid="stMetricValue"] {color:#e8f5ff!important;font-family:'Space Grotesk',sans-serif!important;font-size:1.7rem!important;}
[data-testid="stMetricDelta"] {font-size:.82rem!important;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#24374d!important;}
.stButton>button[kind="primary"], .stFormSubmitButton>button {background:#36d9c2!important;border:1px solid #36d9c2!important;color:#06141c!important;font-weight:700!important;border-radius:10px!important;min-height:46px;}
.stButton>button {border-radius:10px!important;}
[data-baseweb="input"] > div,[data-baseweb="select"] > div {background:#111f30!important;border-color:#263a51!important;}
[data-testid="stForm"] {background:#0d1725;border:1px solid #24374d;border-radius:16px;padding:22px;}
[data-testid="stTabs"] button {font-weight:600;}
[data-testid="stAlert"] {border-radius:12px;}
hr {border-color:#203149!important;}
.brand {display:flex;align-items:center;gap:12px;margin-bottom:22px;}
.brand-mark {width:39px;height:39px;background:#153c42;border:1px solid #2a8d8b;color:#42e3d0;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:25px;font-weight:700;}
.brand-name {font-family:'Space Grotesk',sans-serif;color:#f4fbff;font-weight:700;font-size:21px;letter-spacing:-.04em;line-height:1.1;}
.brand-sub {font-size:10px;color:#7793ad;letter-spacing:.14em;font-weight:700;}
.eyebrow {color:#46d8c7;text-transform:uppercase;letter-spacing:.19em;font-size:11px;font-weight:800;margin-bottom:7px;}
.hero {position:relative;overflow:hidden;background:linear-gradient(118deg,#11243a 0%,#0e1d30 60%,#0d2a35 100%);border:1px solid #274158;border-radius:19px;padding:30px 32px;margin:8px 0 23px;}
.hero:before {content:"";position:absolute;right:-95px;top:-125px;width:400px;height:400px;border:1px solid #28556b;border-radius:50%;box-shadow:0 0 0 65px #17344744,0 0 0 130px #17344722;pointer-events:none;}
.hero h1 {font-size:37px!important;line-height:1.12!important;margin:8px 0 10px!important;max-width:720px;}
.hero p {color:#a6bdd1;font-size:15px;max-width:660px;line-height:1.65;margin-bottom:0;}
.pill {display:inline-block;border:1px solid #2e5864;border-radius:99px;padding:6px 10px;background:#10343a;color:#66e7d4;font-size:11px;font-weight:700;letter-spacing:.06em;}
.panel {background:#0f1a2a;border:1px solid #21354c;border-radius:15px;padding:19px 21px;margin:6px 0 17px;}
.panel-title {font-family:'Space Grotesk',sans-serif;color:#eff7ff;font-size:16px;font-weight:700;margin-bottom:7px;}
.panel-copy {color:#9cb0c5;font-size:13px;line-height:1.65;}
.label {font-size:11px;color:#7995b0;text-transform:uppercase;letter-spacing:.11em;font-weight:700;}
.big-number {font-family:'Space Grotesk',sans-serif;font-size:48px;letter-spacing:-.055em;font-weight:700;color:#efffff;line-height:1.2;}
.accent {color:#4ce3d0;}
.divider-line {height:1px;background:#24374a;margin:17px 0;}
.mini-row {display:flex;justify-content:space-between;gap:10px;padding:9px 0;border-bottom:1px solid #1d3043;color:#a9bed2;font-size:13px;}
.mini-row strong {color:#ecf7ff;}
.footer-note {color:#708da5;font-size:11px;margin-top:20px;}
.section-label {font-family:'Space Grotesk',sans-serif;font-size:19px;color:#edf6ff;font-weight:700;margin:19px 0 6px;}
[data-testid="stDataFrame"] {border:1px solid #253a50;border-radius:10px;overflow:hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model(path: str, mtime: float):
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_json(path: str, mtime: float):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def artifacts():
    missing = [p.name for p in [MODEL_PATH, FEATURES_PATH, METRICS_PATH] if not p.is_file()]
    model, features, metrics, issue = None, FEATURES, None, None
    if not missing:
        try:
            model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            features = load_json(str(FEATURES_PATH), FEATURES_PATH.stat().st_mtime)
            metrics = load_json(str(METRICS_PATH), METRICS_PATH.stat().st_mtime)
            if not isinstance(features, list) or len(features) != 17 or set(features) != set(FEATURES):
                raise ValueError("Feature metadata must contain the expected 17 features.")
            if getattr(model, "n_features_in_", None) != len(features):
                raise ValueError("Saved model feature count does not match metadata.")
        except Exception as exc:
            model, issue = None, str(exc)
    return model, features, metrics, missing, issue

model, feature_names, saved_metrics, missing, model_issue = artifacts()
ready = model is not None

# Exact calculations and input schema preserved from original app.py.
def make_features(timestamp: datetime, current_count: float, lag_1: float, lag_2: float, lag_3: float, lag_24: float, speed_limit: float, frc: float, distance: float, names: list[str]):
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    previous_counts = np.array([lag_1, lag_2, lag_3], dtype=float)
    values = {
        "probeCount": current_count,
        "probeCount_lag_1": lag_1,
        "probeCount_lag_2": lag_2,
        "probeCount_lag_3": lag_3,
        "probeCount_lag_24": lag_24,
        "probeCount_rolling_mean_3h": float(np.mean(previous_counts)),
        "probeCount_rolling_std_3h": float(np.std(previous_counts, ddof=1)),
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
        "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
        "speedLimit": speed_limit,
        "frc": frc,
        "distance": distance,
    }
    frame = pd.DataFrame([[values[n] for n in names]], columns=names)
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("Input features contain invalid values.")
    return frame

COLORS = {"background": "rgba(0,0,0,0)", "paper": "rgba(0,0,0,0)", "grid": "#203449", "text": "#a9bed2", "teal": "#47dfcb", "blue": "#6b9cff", "orange": "#f8b46c"}
def chart_style(fig, height=320, showlegend=True):
    fig.update_layout(height=height, margin=dict(l=10,r=12,t=22,b=15), paper_bgcolor=COLORS["paper"], plot_bgcolor=COLORS["background"], font=dict(family="DM Sans, sans-serif", color=COLORS["text"], size=12), showlegend=showlegend, legend=dict(orientation="h", y=1.12, x=0), hovermode="x unified")
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="#25394d")
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False)
    return fig

def section(title, subtitle=None):
    st.markdown(f'<div class="section-label">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)

def panel(title, body):
    st.markdown(f'<div class="panel"><div class="panel-title">{title}</div><div class="panel-copy">{body}</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="brand"><div class="brand-mark">◈</div><div><div class="brand-name">TRAFFICIQ</div><div class="brand-sub">MOBILITY INTELLIGENCE</div></div></div>', unsafe_allow_html=True)
    page = st.radio("WORKSPACE", ["Command Center", "Road Forecast", "Road Explorer", "Traffic Trends", "Model Intelligence", "Data Explorer", "About the system"], label_visibility="collapsed")
    st.markdown("---")
    if ready:
        st.success("● Prediction engine connected")
    elif missing:
        st.warning("● Preview mode · model offline")
    else:
        st.error("● Model could not be loaded")
    st.caption("Historical research prototype · not live traffic telemetry")
    st.markdown("---")
    st.markdown("**MODEL**")
    st.caption("Random Forest Regression · 100 trees")
    st.markdown("**FORECAST**")
    st.caption("One hour ahead · one road segment")
    st.markdown("**DATASET**")
    st.caption("New Delhi · August 2024")

st.markdown('<div class="eyebrow">NEW DELHI / TRAFFIC ANALYTICS / 2024 DATASET</div>', unsafe_allow_html=True)

if page == "Command Center":
    st.markdown('<div class="hero"><span class="pill">◈ PREDICTIVE ANALYTICS WORKSPACE</span><h1>See the next hour.<br><span class="accent">Before it happens.</span></h1><p>Explore historical traffic-probe patterns, test next-hour forecasts, and understand the model behind every prediction — in one focused command center.</p></div>', unsafe_allow_html=True)
    a,b,c,d = st.columns(4)
    a.metric("ROAD SEGMENTS IN DATASET", "24,938")
    b.metric("FORECAST HORIZON", "1 hour")
    c.metric("MODEL R² · HISTORICAL TEST", "0.9633")
    d.metric("MODEL MAE · PROBE UNITS", "19.78")
    st.caption("Dataset and benchmark statistics from the original project notebook; not live citywide measurements.")
    left,right = st.columns([1.7,1], gap="large")
    with left:
        section("Model performance at a glance", "Final notebook comparison · same 500,000 sampled training rows and held-out chronological test set")
        fig=go.Figure()
        for name, vals in BENCHMARK.items():
            fig.add_trace(go.Bar(name=name, x=[name], y=[vals["MAE"]], marker_color=COLORS["teal"] if name=="Random Forest" else (COLORS["blue"] if name=="Polynomial Regression" else "#3c5872"), text=[f'{vals["MAE"]:.2f}'], textposition="outside", hovertemplate="%{x}<br>MAE: %{y:.2f}<extra></extra>"))
        fig.update_yaxes(title="Mean absolute error · lower is better", range=[0,31])
        st.plotly_chart(chart_style(fig,330,False), use_container_width=True, config={"displayModeBar":False})
    with right:
        section("System status")
        panel("Prediction engine", "<span style='color:#4ce3d0'>Connected — ready for inference</span>" if ready else "<span style='color:#f5ba7a'>Awaiting saved model artifacts</span><br>Dashboard and model explanations are available in preview mode.")
        panel("What the forecast means", "A predicted <strong>probe count for the next hour</strong> on one road segment. It is not an exact number of vehicles or a validated congestion level.")
        if "last_prediction" in st.session_state:
            r=st.session_state["last_prediction"]
            panel("Latest prediction", f'<span style="font-size:27px;color:#4ce3d0;font-weight:700">{r["prediction"]:,.2f}</span><br>Probe count forecast for {r["forecast_time"].strftime("%d %b, %H:%M")}.')
    section("Explore the workspace")
    p1,p2,p3=st.columns(3)
    with p1: panel("01 / Road Forecast", "Select a real Delhi road, review its recorded activity and generate a next-hour forecast using automatically prepared inputs.")
    with p2: panel("02 / Model Intelligence", "Compare three regression models, examine performance metrics, and understand which features matter.")
    with p3: panel("03 / Data Explorer", "Upload an optional CSV of observations to explore its trends, distribution, and columns — no fabricated live data.")
    st.info("Use Road Forecast to select an actual road and observation time. Prepare historical data once using the included script.")


DATA_DIR=ROOT / "trafficiq_data"
@st.cache_data(show_spinner=False)
def load_roads():
    return json.loads((DATA_DIR/"roads.json").read_text(encoding="utf-8"))
@st.cache_data(show_spinner=False)
def load_observations():
    df=pd.read_csv(DATA_DIR/"observations.csv.gz",compression="gzip",dtype={"segment_id":str,"date":str})
    return df

def get_series(segment):
    df=load_observations()
    rows=df.loc[df.segment_id==segment,["date","counts"]]
    points=[]
    for date_text,raw in rows.itertuples(index=False,name=None):
        for hour,count in enumerate(json.loads(raw)):
            if count is not None:
                points.append((pd.Timestamp(date_text)+pd.Timedelta(hours=hour),float(count)))
    if not points:return pd.Series(dtype=float)
    times,values=zip(*sorted(points))
    return pd.Series(values,index=pd.DatetimeIndex(times),name="Probe observations").sort_index()

def road_controls(prefix):
    roads=load_roads()
    query=st.text_input("Find a road by name",placeholder="e.g. Ghata Masjid Road",key=prefix+"query")
    matches=[(sid,r) for sid,r in roads.items() if query.casefold() in str(r.get("name","")).casefold()]
    if not matches: st.info("No matching roads found.");return None,None,None
    matches=sorted(matches,key=lambda item:(str(item[1].get("name","")).lower(),item[0]))
    matches=matches[:300]
    labels={sid:f'{r["name"]} · {sid[-12:]}' for sid,r in matches}
    sid=st.selectbox("Road segment",list(labels),format_func=lambda v:labels[v],key=prefix+"sid")
    return sid,roads[sid],get_series(sid)

def draw_map(road):
    coords=road.get("coordinates") or []
    if not coords:
        st.info("No geometry available for this segment.");return
    import plotly.graph_objects as go
    lons=[float(c[0]) for c in coords if len(c)>=2]
    lats=[float(c[1]) for c in coords if len(c)>=2]
    if not lons:return
    fig=go.Figure(go.Scattermap(lat=lats,lon=lons,mode="lines+markers",line=dict(width=6,color="#36d9c2"),marker=dict(size=5,color="#9bf8e8"),hovertemplate="Selected road segment<extra></extra>"))
    fig.update_layout(map=dict(style="carto-darkmatter",center=dict(lat=sum(lats)/len(lats),lon=sum(lons)/len(lons)),zoom=13),height=370,margin=dict(l=0,r=0,t=0,b=0),paper_bgcolor="#0f1a2a")
    st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

if page in ["Road Forecast","Road Explorer","Traffic Trends"]:
    st.title(page)
    if not (DATA_DIR/"roads.json").exists() or not (DATA_DIR/"observations.csv.gz").exists():
        st.warning("Historical data hasn't been prepared yet. Run `prepare_trafficiq_data.py` once in your project folder, then refresh.")
        st.stop()
    st.caption("Historical New Delhi road-probe observations · August 2024 · not live traffic")
    sid,road,series=road_controls(page)
    if sid is not None:
        a,b,c=st.columns(3)
        a.metric("Road name",road.get("name","Unknown"))
        b.metric("Recorded hours",f"{len(series):,}")
        c.metric("Speed limit (dataset)",str(road.get("speedLimit") or "Unknown"))
        if page=="Road Explorer":
            st.subheader("Selected road location")
            draw_map(road)
            with st.expander("Road segment details"):
                st.json({k:v for k,v in road.items() if k!="coordinates"})
        elif page=="Traffic Trends":
            if len(series):
                fig=go.Figure(go.Scatter(x=series.index,y=series.values,mode="lines",line=dict(color="#42dfcd",width=2),fill="tozeroy",fillcolor="rgba(66,223,205,.10)",name="Recorded activity"))
                fig.update_yaxes(title="Probe observations")
                st.plotly_chart(chart_style(fig,430),use_container_width=True)
                st.caption("Probe observations represent dataset sampling activity, not exact vehicle counts.")
        else:
            if series.empty: st.warning("No recorded observations for this road.");st.stop()
            available=series.index.sort_values()
            eligible=[t for t in available if all(t-pd.Timedelta(hours=h) in series.index for h in (1,2,3,24))]
            if not eligible: st.warning("No timestamps have all required 1, 2, 3 and 24-hour historical inputs.");st.stop()
            selected=st.select_slider("Observation date and hour",options=list(eligible),value=eligible[-1],format_func=lambda t:t.strftime("%d %b %Y · %I:%M %p"))
            current=float(series.loc[selected]);previous=[float(series.loc[selected-pd.Timedelta(hours=h)]) for h in (1,2,3,24)]
            next_t=selected+pd.Timedelta(hours=1)
            historical_actual=float(series.loc[next_t]) if next_t in series.index else None
            left,right=st.columns([1.7,1],gap="large")
            with left:
                st.subheader("Recent activity & forecast")
                recent=series.loc[(series.index>=selected-pd.Timedelta(hours=12)) & (series.index<=selected)]
                fig=go.Figure(go.Scatter(x=recent.index,y=recent.values,mode="lines+markers",line=dict(color="#45dccc",width=3),name="Recorded activity"))
                fig.update_yaxes(title="Probe observations")
                st.plotly_chart(chart_style(fig,330),use_container_width=True)
            with right:
                st.subheader("Road location")
                draw_map(road)
            st.metric("Activity at selected hour",f"{current:,.0f} probe observations")
            if ready:
                if st.button("Generate next-hour forecast",type="primary",use_container_width=True):
                    try:
                        vals=[road.get(k) for k in ("speedLimit","frc","distance")]
                        if any(v is None for v in vals): raise ValueError("Road characteristics are missing for this segment")
                        frame=make_features(selected.to_pydatetime(),current,previous[0],previous[1],previous[2],previous[3],*map(float,vals),feature_names)
                        prediction=float(model.predict(frame)[0])
                        st.session_state["road_forecast"]={"sid":sid,"timestamp":selected,"prediction":prediction,"actual":historical_actual,"current":current}
                    except Exception as exc:st.error(f"Forecast unavailable: {exc}")
                result=st.session_state.get("road_forecast")
                if result and result["sid"]==sid and result["timestamp"]==selected:
                    pred=result["prediction"]
                    st.success(f"Forecast for {next_t.strftime('%d %b %Y, %I:%M %p')}: {pred:,.1f} probe observations")
                    change=(pred/current-1)*100 if current else None
                    if change is not None:st.metric("Change vs selected hour",f"{change:+.1f}%")
                    if historical_actual is not None:st.info(f"Recorded value for that next hour: {historical_actual:,.0f} probe observations. This is a historical comparison, not a live forecast.")
                    st.caption("This predicts probe counts, not exact vehicles, congestion severity or travel time.")
                    forecast_fig=go.Figure()
                    forecast_fig.add_trace(go.Scatter(x=recent.index,y=recent.values,mode="lines+markers",name="Observed",line=dict(color="#42dfcd",width=3)))
                    forecast_fig.add_trace(go.Scatter(x=[selected,next_t],y=[current,pred],mode="lines+markers",name="Predicted",line=dict(color="#4cf08a",width=3,dash="dash")))
                    if historical_actual is not None:forecast_fig.add_trace(go.Scatter(x=[next_t],y=[historical_actual],mode="markers",marker=dict(size=12,color="#f4b66c"),name="Actual next hour"))
                    st.plotly_chart(chart_style(forecast_fig,350),use_container_width=True)
            else:st.warning("The local Random Forest model is not connected. Check the three files in saved_model.")

elif page == "Manual Prediction Lab":
    st.title("Prediction Lab")
    st.caption("Create a one-hour-ahead forecast from observations for a single road segment. Inputs below are illustrative defaults, not live sensor readings.")
    if not ready:
        st.warning("Prediction engine unavailable. You can explore the form, but generating a forecast requires the saved model files.")
        if missing: st.caption("Missing in saved_model/: " + ", ".join(missing))
        if model_issue: st.caption("Model loading error: " + model_issue)
    with st.form("predict_form"):
        st.markdown("#### 01 · Observation context")
        col1,col2=st.columns(2)
        with col1: observation_date=st.date_input("Observation date", value=date.today())
        with col2: observation_time=st.time_input("Observation hour", value=time(12,0))
        st.markdown("#### 02 · Probe-count history")
        st.caption("All five counts should describe the same road segment. The three-hour rolling features are calculated automatically.")
        c1,c2,c3=st.columns(3)
        with c1:
            current=st.number_input("Current hour", min_value=0.0, value=100.0, step=1.0)
            lag1=st.number_input("1 hour ago", min_value=0.0, value=95.0, step=1.0)
        with c2:
            lag2=st.number_input("2 hours ago", min_value=0.0, value=90.0, step=1.0)
            lag3=st.number_input("3 hours ago", min_value=0.0, value=85.0, step=1.0)
        with c3: lag24=st.number_input("24 hours ago", min_value=0.0, value=80.0, step=1.0)
        st.markdown("#### 03 · Road characteristics")
        st.caption("Use values in the original dataset's units and coding; changing their meaning can make forecasts unreliable.")
        r1,r2,r3=st.columns(3)
        with r1: speed_limit=st.number_input("Speed limit · dataset units", min_value=0.0, value=50.0, step=1.0)
        with r2: frc=st.number_input("Functional road class (FRC)", min_value=0.0, value=3.0, step=1.0)
        with r3: distance=st.number_input("Road-segment distance · dataset units", min_value=0.0, value=1.0, step=0.1)
        submit=st.form_submit_button("Generate next-hour forecast  →", disabled=not ready, use_container_width=True)
    if submit and ready:
        try:
            timestamp=datetime.combine(observation_date, observation_time)
            df=make_features(timestamp,current,lag1,lag2,lag3,lag24,speed_limit,frc,distance,feature_names)
            predicted=float(model.predict(df)[0])
            if not np.isfinite(predicted): raise ValueError("Model returned a non-finite prediction")
            forecast_time=timestamp+timedelta(hours=1)
            st.session_state["last_prediction"]={"prediction":predicted,"timestamp":timestamp,"forecast_time":forecast_time,"current":current,"lags":[lag3,lag2,lag1,current],"df":df}
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
    if "last_prediction" in st.session_state:
        r=st.session_state["last_prediction"]
        section("Forecast result", "Real model inference from your supplied inputs")
        a,b,c=st.columns(3)
        a.metric("NEXT-HOUR PROBE COUNT",f'{r["prediction"]:,.2f}')
        b.metric("CURRENT OBSERVATION",f'{r["current"]:,.2f}')
        diff=r["prediction"]-r["current"]
        c.metric("CHANGE VS CURRENT",f'{diff:+,.2f}', delta=f'{diff:+,.2f} probe units')
        st.caption(f'Observation: {r["timestamp"].strftime("%d %b %Y, %H:%M")}  ·  Forecast for: {r["forecast_time"].strftime("%d %b %Y, %H:%M")}')
        fig=go.Figure()
        ts=[r["timestamp"]-timedelta(hours=i) for i in [3,2,1,0]]
        fig.add_trace(go.Scatter(x=ts,y=r["lags"],mode="lines+markers",name="Entered observations",line=dict(color=COLORS["blue"],width=3),marker=dict(size=9)))
        fig.add_trace(go.Scatter(x=[ts[-1],r["forecast_time"]],y=[r["current"],r["prediction"]],mode="lines+markers",name="Model forecast",line=dict(color=COLORS["teal"],width=3,dash="dash"),marker=dict(size=11)))
        fig.update_yaxes(title="Probe-count observations")
        st.plotly_chart(chart_style(fig,370),use_container_width=True,config={"displayModeBar":False})
        st.info("The line shows the four recent values you entered plus one model forecast. It is not a continuous forecast or live traffic feed. A higher probe count does not necessarily mean higher congestion.")
        with st.expander("Inspect all 17 features used for this prediction"):
            st.dataframe(r["df"].T.rename(columns={0:"Value"}),use_container_width=True)
        export=pd.DataFrame([{"observation_time":r["timestamp"].isoformat(),"forecast_time":r["forecast_time"].isoformat(),"current_probe_count":r["current"],"predicted_next_hour_probe_count":r["prediction"]}])
        st.download_button("Download prediction CSV",export.to_csv(index=False),file_name="trafficiq_prediction.csv",mime="text/csv")

elif page == "Model Intelligence":
    st.title("Model Intelligence")
    st.caption("Inspect the project's actual final model-comparison results and learn how predictions are constructed.")
    a,b,c=st.columns(3)
    a.metric("BEST MODEL", "Random Forest")
    b.metric("BEST TEST R²", "0.9633")
    c.metric("TRAINING SAMPLE / MODEL", "500,000")
    section("Compare algorithms", "Results recorded in the project notebook, not calculated from the current inputs")
    metric=st.segmented_control("Metric",["MAE","RMSE","R2"],default="MAE",format_func=lambda x: "R²" if x=="R2" else x)
    fig=go.Figure()
    for name,v in BENCHMARK.items():
        fig.add_trace(go.Bar(x=[name],y=[v[metric]],name=name,marker_color=COLORS["teal"] if name=="Random Forest" else COLORS["blue"] if name=="Polynomial Regression" else "#415b77",text=[f'{v[metric]:.4f}'],textposition="outside"))
    fig.update_yaxes(title=f'{metric} · '+("higher is better" if metric=="R2" else "lower is better"),range=[0,max(v[metric] for v in BENCHMARK.values())*1.22])
    st.plotly_chart(chart_style(fig,340,False),use_container_width=True,config={"displayModeBar":False})
    table=pd.DataFrame(BENCHMARK).T.rename(columns={"R2":"R²"})
    st.dataframe(table.style.format({"MAE":"{:.4f}","RMSE":"{:.4f}","R²":"{:.4f}"}),use_container_width=True)
    section("Feature intelligence")
    l,r=st.columns([1.5,1],gap="large")
    with l:
        if ready and hasattr(model,"feature_importances_") and len(model.feature_importances_)==len(feature_names):
            importance=pd.Series(model.feature_importances_,index=feature_names).sort_values().tail(10)
            fig=go.Figure(go.Bar(x=importance.values,y=importance.index,orientation="h",marker_color=COLORS["teal"]))
            fig.update_xaxes(title="Random Forest impurity-based importance")
            st.plotly_chart(chart_style(fig,390,False),use_container_width=True,config={"displayModeBar":False})
            st.caption("Calculated from the loaded Random Forest. Importance is not causation.")
        else:
            panel("Feature importance awaiting model", "When the saved Random Forest is connected, this panel will read its actual feature_importances_ values. No invented values are displayed.")
    with r:
        panel("What the model predicts", "The numerical <strong>next-hour probe count</strong> for the same road segment, using 17 engineered inputs.")
        panel("Why Random Forest?", "It achieved the lowest MAE and RMSE and highest R² among the three compared algorithms on the held-out test set.")
        panel("Important limitation", "The historical test covers a limited period. The reported R² is explained variation, not classification accuracy.")
    with st.expander("Understand the evaluation metrics"):
        st.markdown("**MAE** — mean absolute prediction error, in probe-count units. **RMSE** — square-root of mean squared error; penalizes larger misses more strongly. **R²** — improvement in squared-error terms over predicting the mean of the target.")
    with st.expander("View the 17 model input features"):
        st.dataframe(pd.DataFrame({"Feature name":FEATURES}),hide_index=True,use_container_width=True)
    if saved_metrics is not None:
        with st.expander("Inspect saved model metrics file"):
            st.json(saved_metrics)

elif page == "Data Explorer":
    st.title("Data Explorer")
    st.caption("Inspect real CSV data you upload. This page does not invent live traffic observations or silently load a fabricated dataset.")
    a,b,c=st.columns(3)
    a.metric("DATASET ROAD SEGMENTS", "24,938")
    b.metric("FINAL FEATURE-READY ROWS", "11,271,976")
    c.metric("MODEL INPUT FEATURES", "17")
    panel("About the dataset", "New Delhi Traffic Probe Count & Analytics Dataset (2024). The original notebook builds hourly road-segment features from historical probe-count observations. The original raw source is GeoJSON; this explorer accepts a <strong>CSV extract</strong> for convenient visualization.")
    upload=st.file_uploader("Upload a CSV extract to explore",type=["csv"],help="For large data, export a smaller CSV sample first. Files are read locally by this Streamlit session.")
    if upload is not None:
        try:
            data=pd.read_csv(upload,nrows=100000)
            if data.empty: st.warning("The uploaded CSV contains no data rows.")
            else:
                st.caption("For responsiveness, up to the first 100,000 rows are loaded.")
                a,b,c=st.columns(3)
                a.metric("ROWS LOADED",f"{len(data):,}")
                b.metric("COLUMNS",len(data.columns))
                c.metric("MISSING CELLS",f"{int(data.isna().sum().sum()):,}")
                st.dataframe(data.head(100),use_container_width=True)
                numeric=data.select_dtypes(include="number").columns.tolist()
                if numeric:
                    col=st.selectbox("Explore a numerical column",numeric)
                    vals=data[col].dropna()
                    if len(vals):
                        fig=go.Figure(go.Histogram(x=vals,nbinsx=45,marker_color=COLORS["teal"]))
                        fig.update_xaxes(title=col)
                        fig.update_yaxes(title="Number of rows")
                        st.plotly_chart(chart_style(fig,340,False),use_container_width=True,config={"displayModeBar":False})
                else: st.info("No numerical columns found for histogram visualization.")
        except Exception as exc: st.error(f"Could not read the uploaded CSV: {exc}")
    else:
        st.info("No CSV uploaded yet. The dataset statistics above come from the original project notebook; charts will appear only after you provide actual CSV data.")
    with st.expander("What can I upload?"):
        st.write("A CSV with hourly probe counts, timestamps, road-segment identifiers, or engineered features. Export a sample from the notebook if your original dataset is too large.")

else:
    st.title("About TRAFFICIQ")
    st.caption("A research prototype with a product-minded interface — not a deployed municipal traffic system.")
    panel("The problem", "Historical traffic datasets contain patterns that are hard to inspect from raw tables alone. Forecasting the next observation can support exploration and future decision-support tools.")
    panel("The implemented solution", "A Random Forest Regression model trained on historical New Delhi road-segment probe counts. It uses 17 engineered inputs to forecast the next-hour probe count for the same segment.")
    section("How a prediction works")
    st.markdown("**1. Supply observations** → **2. Create lag, rolling and cyclical time features** → **3. Match saved feature order** → **4. Run saved Random Forest** → **5. Visualize next-hour forecast**")
    section("What's real vs what's on the roadmap")
    st.dataframe(pd.DataFrame([
        {"Capability":"Next-hour probe-count prediction","Status":"Implemented; requires saved model"},
        {"Capability":"Model comparison and metrics","Status":"Implemented from notebook results"},
        {"Capability":"Interactive input/forecast chart","Status":"Implemented"},
        {"Capability":"CSV exploratory analytics","Status":"Implemented"},
        {"Capability":"Live traffic API integration","Status":"Future scope"},
        {"Capability":"Geospatial road map","Status":"Future scope; requires geometry"},
        {"Capability":"Validated congestion categories","Status":"Future scope; requires suitable labels"},
        {"Capability":"Traffic alerts and route recommendations","Status":"Future scope"},
    ]),hide_index=True,use_container_width=True)
    section("Research limitations")
    st.markdown("- Probe count is not necessarily the number of vehicles, speed, or congestion.\n- The evaluation covers a limited historical time window; real-world seasonal performance is unverified.\n- User-entered examples are not live observations.\n- Model artifacts should come from your trusted project partner; never load an untrusted `.joblib` file.")
    section("Model files expected")
    st.code("saved_model/\n  random_forest_model.joblib\n  model_features.json\n  model_metrics.json",language="text")
    if missing: st.warning("Currently missing: " + ", ".join(missing))
    elif model_issue: st.error("Model loading error: " + model_issue)
    else: st.success("All saved artifacts loaded successfully.")

st.markdown('<div class="footer-note">TRAFFICIQ / STATISTICAL MACHINE LEARNING PROJECT &nbsp;·&nbsp; HISTORICAL DATA PROTOTYPE &nbsp;·&nbsp; NOT LIVE TRAFFIC INFORMATION</div>', unsafe_allow_html=True)
