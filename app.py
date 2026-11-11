"""NISAR Disaster Detection Platform — Streamlit app.

Run:  streamlit run app.py
"""

from __future__ import annotations

import io
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.ndimage import gaussian_filter
from sklearn.metrics import f1_score, precision_score, recall_score

from src.coherence import CoherenceAnalyzer
from src.inference import DisasterDetector
from src.model import ModelTrainer
from src.physics import SARPhysics
from src.visualization import DisasterMap

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.path.join(APP_DIR, "data", "sample", "unet_checkpoint.pt")
# Demo AOI: northern Tien Shan foothills near Almaty (landslide-prone)
DEMO_BOUNDS = (43.00, 43.40, 76.70, 77.20)  # lat_min, lat_max, lon_min, lon_max

st.set_page_config(page_title="NISAR Disaster Detection Platform", page_icon=":material/satellite_alt:",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
div[data-testid="stMetric"] {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border: 1px solid rgba(255,75,75,0.25); border-radius: 12px; padding: 14px 18px;
}
.risk-badge {display:inline-block; padding:10px 26px; border-radius:30px; font-size:26px;
  font-weight:800; letter-spacing:2px; color:#fff;}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800;900&family=Source+Serif+4:wght@400;600;700&display=swap');

/* ── БАЗОВЫЕ ШРИФТЫ ── */
h1, h2, h3, h4, h5, h6,
h1 *, h2 *, h3 *, h4 *, h5 *, h6 *,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
}

body, p, span, div, label,
.stMarkdown p, .stMarkdown li {
    font-family: 'Source Serif 4', Georgia, serif !important;
}

/* ── САЙДБАР ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 700 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    font-family: 'Source Serif 4', Georgia, serif !important;
}

/* ── КНОПКИ (текст лежит во вложенном <p>) ── */
.stButton > button,
.stButton > button p,
.stDownloadButton > button p {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* ── ВКЛАДКИ ── */
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}

/* ── МЕТРИКИ ── */
[data-testid="stMetricLabel"] p {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 900 !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploaderDropzone"] small {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: #888 !important;
    font-size: 13px !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] summary p {
    font-family: 'Playfair Display', Georgia, serif !important;
    font-weight: 700 !important;
}

/* ── ИКОНКИ: последним и с максимальной специфичностью ──
   Streamlit рисует иконки лигатурами шрифта Material Symbols Rounded:
   текст "keyboard_double_arrow_left", "upload", "expand_more" превращается
   в глиф только этим шрифтом. Любой другой шрифт показывает слово как текст. */
html body span[data-testid="stIconMaterial"][data-testid],
html body [data-testid="stSidebar"] span[data-testid="stIconMaterial"],
html body [data-testid="stExpander"] summary span[data-testid="stIconMaterial"],
html body [data-testid="stFileUploader"] span[data-testid="stIconMaterial"],
html body [class*="material-symbols"][class] {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: 400 !important;
    font-style: normal !important;
    letter-spacing: normal !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demo(seed: int):
    return SARPhysics.generate_synthetic_gunw(shape=(512, 512), seed=seed)


def _norm01(a):
    a = np.nan_to_num(np.asarray(a, dtype=np.float32))
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)


def load_uploaded(file) -> dict:
    """Load .npz (keys below) or NISAR/ARIA GUNW HDF5/NetCDF (needs h5py)."""
    name = file.name.lower()
    raw = file.getvalue()
    if name.endswith(".npz"):
        z = np.load(io.BytesIO(raw))
        phase = z["unwrapped_phase"]
        coh_after = z["coherence_after"] if "coherence_after" in z else z["coherence"]
        coh_before = z["coherence_before"] if "coherence_before" in z else None
        amp = z["amplitude"] if "amplitude" in z else None
        dem = z["dem"] if "dem" in z else None
    else:
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError("Reading HDF5/NetCDF GUNW needs `pip install h5py`, "
                               "or upload an .npz file.") from e
        found = {}
        with h5py.File(io.BytesIO(raw), "r") as f:
            def visit(path, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
                    key = path.split("/")[-1].lower()
                    found.setdefault(key, obj[()])
            f.visititems(visit)
        pick = lambda *keys: next((found[k] for k in found for q in keys if q in k), None)
        phase = pick("unwrappedphase", "unwrapped_phase")
        coh_after = pick("coherencemagnitude", "coherence")
        amp = pick("amplitude", "magnitude")
        dem = pick("dem", "height")
        coh_before = None
        if phase is None or coh_after is None:
            raise RuntimeError("Could not find unwrapped phase / coherence datasets in file.")

    phase = np.nan_to_num(phase).astype(np.float32)
    coh_after = np.clip(np.nan_to_num(coh_after), 0, 1).astype(np.float32)
    if coh_before is None:
        # Single-pair product: use a smoothed high-percentile coherence as pre-event reference
        coh_before = np.maximum(coh_after, gaussian_filter(coh_after, 15) + 0.1).clip(0, 1)
    amp = _norm01(amp) if amp is not None else _norm01(coh_after)
    dem = dem if dem is not None else np.zeros_like(phase)
    return {
        "unwrapped_phase": phase,
        "wrapped_phase": SARPhysics.wrap_phase(phase),
        "coherence_before": coh_before.astype(np.float32),
        "coherence_after": coh_after,
        "amplitude": amp.astype(np.float32),
        "dem": np.nan_to_num(dem).astype(np.float32),
        "truth_mask": None,
        "anomalies": [],
    }


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def train_model(data: dict, epochs: int, lr: float):
    disp = SARPhysics.phase_to_displacement(data["unwrapped_phase"], st.session_state.wavelength)
    dcoh = CoherenceAnalyzer.compute_coherence_change(data["coherence_before"], data["coherence_after"])
    labels = ModelTrainer.create_pseudo_labels(disp, dcoh, coh_thr=st.session_state.coh_thr * 0.7)
    trainer = ModelTrainer()
    stack = np.stack([data["coherence_before"], data["coherence_after"], data["amplitude"]])
    trainer.prepare_tiles(stack, disp, tile_size=256, overlap=32, labels=labels)
    trainer.train(epochs=epochs, lr=lr)
    trainer.save_checkpoint(CHECKPOINT)
    return trainer.history


def run_analysis(data: dict, wavelength: float, inc_angle: float, coh_thr: float,
                 bounds, min_area: int):
    los = SARPhysics.phase_to_displacement(data["unwrapped_phase"], wavelength)
    vert = SARPhysics.compute_los_to_vertical(los, inc_angle)
    dcoh = CoherenceAnalyzer.compute_coherence_change(data["coherence_before"], data["coherence_after"])
    coh_mask = CoherenceAnalyzer.threshold_damage_zones(dcoh, coh_thr)
    risk_map = CoherenceAnalyzer.compute_risk_score(vert, dcoh)

    detector = DisasterDetector(checkpoint_path=CHECKPOINT)
    if detector.is_trained:
        prob = detector.ensemble_predict(data["coherence_before"], data["coherence_after"],
                                         data["amplitude"])
    else:
        # Fallback before first training: physics-based probability from risk map
        prob = gaussian_filter(np.clip((risk_map - 30) / 40, 0, 1), 2)
    ml_mask = detector.postprocess(prob, min_area_px=min_area)
    events = detector.extract_event_polygons(ml_mask, bounds, vert, dcoh)

    # Reference labels for metrics: synthetic truth if available, else pseudo-labels
    if data.get("truth_mask") is not None:
        ref, ref_name = data["truth_mask"], "synthetic ground truth"
    else:
        ref = ModelTrainer.create_pseudo_labels(vert, dcoh).astype(bool)
        ref_name = "pseudo-labels"
    y_true, y_pred = ref.ravel(), ml_mask.ravel()
    metrics = dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        ref_name=ref_name,
    )
    return dict(los=los, vert=vert, dcoh=dcoh, coh_mask=coh_mask, risk_map=risk_map,
                prob=prob, ml_mask=ml_mask, events=events, metrics=metrics,
                model_used=detector.is_trained, bounds=bounds)


def synth_timeseries(events, n=15, seed=0):
    """Simulated cumulative displacement history (12-day NISAR repeat) per event."""
    rng = np.random.default_rng(seed)
    end = date(2025, 9, 30)
    dates = [end - timedelta(days=12 * (n - 1 - i)) for i in range(n)]
    series = {}
    t = np.linspace(0, 1, n)
    for f in events["features"][:6]:
        p = f["properties"]
        final = p.get("max_subsidence_mm", -50)
        onset = rng.uniform(0.3, 0.7)
        curve = final * (0.15 * t + 0.85 / (1 + np.exp(-(t - onset) * 14)))
        curve -= curve[0]
        series[p["event_id"]] = curve + rng.normal(0, abs(final) * 0.015, n)
    return dates, series


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def image_fig(arr, title, cmap, zmin=None, zmax=None, cbar_title=""):
    fig = px.imshow(arr, color_continuous_scale=cmap, zmin=zmin, zmax=zmax, aspect="equal")
    fig.update_layout(title=title, template="plotly_dark", height=380,
                      margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      coloraxis_colorbar=dict(title=cbar_title, thickness=12))
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def hist_fig(arr, color):
    sub = np.asarray(arr).ravel()[::7]
    fig = go.Figure(go.Histogram(x=sub, nbinsx=60, marker_color=color, opacity=0.85))
    fig.update_layout(template="plotly_dark", height=200, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      bargap=0.02, showlegend=False)
    return fig


def overlay_rgb(amplitude, mask, prob=None):
    base = np.repeat(_norm01(amplitude)[..., None], 3, axis=2) * 0.85
    color = np.array([1.0, 0.29, 0.29])
    alpha = mask.astype(np.float32)[..., None] * 0.55
    rgb = base * (1 - alpha) + color * alpha
    return (rgb * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("NISAR Disaster Platform")
    st.caption("SAR interferometry · landslides & subsidence")

    mode = st.radio("Mode", ["Demo (Synthetic Data)", "Upload GUNW File"])
    uploaded = None
    seed = 42
    if mode == "Upload GUNW File":
        uploaded = st.file_uploader("GUNW product (.h5 / .nc / .npz)", type=["h5", "nc", "npz"])
        st.caption("`.npz` keys: unwrapped_phase, coherence_after (+ optional coherence_before, "
                   "amplitude, dem)")
    else:
        seed = st.number_input("Scene seed", 0, 9999, 42, help="Changes the synthetic scene")

    st.subheader("Sensor parameters")
    wavelength = st.number_input("Wavelength (m)", 0.01, 0.5, 0.24, 0.01, format="%.3f",
                                 help="NISAR L-band ≈ 0.24 m, S-band ≈ 0.094 m")
    inc_angle = st.slider("Incidence angle (°)", 20.0, 50.0, 35.0, 0.5)

    st.subheader("Detection")
    coh_thr = st.slider("Coherence-loss threshold", 0.1, 0.6, 0.3, 0.05)
    min_area = st.slider("Min event area (px)", 10, 500, 50, 10)

    with st.expander("Training settings"):
        epochs = st.slider("Epochs", 5, 60, 20)
        lr = st.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)

    run_btn = st.button("Run Analysis", type="primary", width="stretch")
    train_btn = st.button("Train Model", width="stretch")

    st.caption("Trained model found" if os.path.exists(CHECKPOINT)
               else "No trained model yet — physics baseline is used until you train.")

st.session_state.wavelength = wavelength
st.session_state.coh_thr = coh_thr

# Load data
data = None
if mode == "Upload GUNW File":
    if uploaded is not None:
        try:
            data = load_uploaded(uploaded)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
    if data is None:
        st.title("NISAR Disaster Detection Platform")
        st.info("Upload a GUNW file in the sidebar, or switch to Demo mode.")
        st.stop()
else:
    data = load_demo(int(seed))

if train_btn:
    with st.sidebar:
        st.markdown("#### Training U-Net")
        hist = train_model(data, epochs, lr)
        st.success(f"Done. Final loss {hist[-1]:.4f}")
    run_btn = True

params = (mode, int(seed), getattr(uploaded, "name", None), wavelength, inc_angle, coh_thr, min_area)
if run_btn or "results" not in st.session_state or st.session_state.get("params") != params:
    with st.spinner("Running InSAR analysis pipeline…"):
        st.session_state.results = run_analysis(data, wavelength, inc_angle, coh_thr,
                                                DEMO_BOUNDS, min_area)
        st.session_state.params = params
R = st.session_state.results

# ---------------------------------------------------------------------------
# Header + tabs
# ---------------------------------------------------------------------------
st.markdown("""
<h1 style="
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: 52px;
  font-weight: 900 !important;
  letter-spacing: -1.5px !important;
  line-height: 1.1 !important;
  color: #ffffff;
  margin-bottom: 4px;
  padding: 0;
">NISAR Disaster Detection Platform</h1>
<p style="
  font-family: 'Source Serif 4', Georgia, serif;
  font-size: 16px;
  color: #888;
  font-style: italic;
  margin-top: 0;
">From NISAR GUNW interferograms to actionable landslide &amp; subsidence risk maps</p>
<hr style="border-color: rgba(255,75,75,0.3); margin: 16px 0 24px;">
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Raw Data", "3D Displacement Map", "ML Detection", "Risk Assessment", "Our Team"])

# --- Tab 1: Raw data --------------------------------------------------------
with tab1:
    layers = [
        ("Wrapped phase (rad)", data["wrapped_phase"], "Twilight", "#9b59b6", "rad"),
        ("Amplitude", data["amplitude"], "gray", "#bdc3c7", ""),
        ("Coherence (post-event)", data["coherence_after"], "Inferno", "#ff8c42", "γ"),
    ]
    cols = st.columns(3)
    for col, (title, arr, cmap, hc, unit) in zip(cols, layers):
        with col:
            st.plotly_chart(image_fig(arr, title, cmap, cbar_title=unit), width="stretch")
            st.plotly_chart(hist_fig(arr, hc), width="stretch")
            a, b, c = st.columns(3)
            a.metric("min", f"{np.nanmin(arr):.3f}")
            b.metric("max", f"{np.nanmax(arr):.3f}")
            c.metric("mean", f"{np.nanmean(arr):.3f}")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        lim = float(np.abs(R["vert"]).max())
        st.plotly_chart(image_fig(R["vert"], "Vertical displacement (mm)", "RdBu",
                                  -lim, lim, "mm"), width="stretch")
    with c2:
        st.plotly_chart(image_fig(R["dcoh"], "Coherence loss Δγ = before − after", "Magma",
                                  0, max(0.6, float(R["dcoh"].max())), "Δγ"),
                        width="stretch")

# --- Tab 2: 3D map ----------------------------------------------------------
with tab2:
    px_km2 = DisasterDetector.pixel_area_km2(R["ml_mask"].shape, R["bounds"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Disaster area", f"{R['ml_mask'].sum() * px_km2:.1f} km²")
    m2.metric("Max subsidence", f"{R['vert'].min():.1f} mm")
    m3.metric("Detected events", len(R["events"]["features"]))
    m4.metric("Max uplift", f"{R['vert'].max():.1f} mm")

    deck = DisasterMap().create_3d_map(R["vert"], R["ml_mask"], data["dem"], R["bounds"],
                                       events=R["events"])
    st.pydeck_chart(deck, width="stretch", height=680)
    st.caption("Heatmap: |vertical displacement| · Columns: detected disaster pixels "
               "(height ∝ subsidence) · Outlines: event risk level (green / yellow / orange / red). "
               "Right-drag to tilt / rotate.")

# --- Tab 3: ML detection ----------------------------------------------------
with tab3:
    if not R["model_used"]:
        st.warning("No trained U-Net yet — showing the physics-based baseline. "
                   "Click **Train Model** in the sidebar (≈10–30 s).")
    met = R["metrics"]
    a, b, c, d = st.columns(4)
    a.metric("Precision", f"{met['precision']:.3f}")
    b.metric("Recall", f"{met['recall']:.3f}")
    c.metric("F1-score", f"{met['f1']:.3f}")
    d.metric("Model", "U-Net + TTA×4" if R["model_used"] else "Physics baseline")
    st.caption(f"Metrics vs. {met['ref_name']}.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(image_fig(R["ml_mask"].astype(float), "Disaster mask", "Reds", 0, 1),
                        width="stretch")
    with c2:
        st.plotly_chart(image_fig(R["prob"], "Model confidence", "Viridis", 0, 1, "p"),
                        width="stretch")
    with c3:
        fig = px.imshow(overlay_rgb(data["amplitude"], R["ml_mask"]))
        fig.update_layout(title="Mask overlay on amplitude", template="plotly_dark", height=380,
                          margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        st.plotly_chart(fig, width="stretch")

    with st.expander("Coherence-threshold mask (non-ML) for comparison"):
        st.plotly_chart(image_fig(R["coh_mask"].astype(float),
                                  f"Δγ > {coh_thr:.2f}", "Oranges", 0, 1),
                        width="stretch")

# --- Tab 4: Risk assessment -------------------------------------------------
with tab4:
    feats = R["events"]["features"]
    overall = max((f["properties"]["risk_score"] for f in feats), default=0.0)
    level = CoherenceAnalyzer.risk_level(overall)
    rgb = CoherenceAnalyzer.level_color(level)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Overall Risk Score", f"{overall:.1f} / 100",
                  delta=f"{len(feats)} active event(s)", delta_color="inverse")
        st.markdown(
            f"<div class='risk-badge' style='background:rgb({rgb[0]},{rgb[1]},{rgb[2]})'>"
            f"● {level}</div>", unsafe_allow_html=True)
        st.caption("LOW < 30 · MEDIUM 30–60 · HIGH 60–80 · CRITICAL > 80")
    with c2:
        st.markdown("#### Event table")
        st.markdown(DisasterMap.create_risk_panel(R["events"]), unsafe_allow_html=True)

    if feats:
        df = pd.DataFrame([f["properties"] for f in feats])
        cols = [c for c in ["event_id", "risk_level", "risk_score", "lat", "lon", "area_km2",
                            "max_subsidence_mm", "mean_displacement_mm", "mean_delta_coh"]
                if c in df]
        with st.expander("Raw event data / download"):
            st.dataframe(df[cols], width="stretch", hide_index=True)
            import json
            st.download_button("Download events GeoJSON", json.dumps(R["events"], indent=2),
                               "nisar_events.geojson", "application/geo+json")

        dates, series = synth_timeseries(R["events"])
        st.plotly_chart(DisasterMap.create_displacement_timeline(dates, series),
                        width="stretch")
        st.caption("Time series are simulated from the final displacement for demo purposes.")

    st.markdown("#### Event Summary")
    if feats:
        top = feats[0]["properties"]
        n_crit = sum(f["properties"]["risk_level"] == "CRITICAL" for f in feats)
        n_high = sum(f["properties"]["risk_level"] == "HIGH" for f in feats)
        total_area = sum(f["properties"].get("area_km2", 0) for f in feats)
        st.info(
            f"The analysis detected **{len(feats)} deformation event(s)** covering "
            f"**{total_area:.1f} km²** ({n_crit} critical, {n_high} high). "
            f"The most severe event **{top['event_id']}** is centred at "
            f"**{top.get('lat')}°N, {top.get('lon')}°E** with a maximum vertical subsidence of "
            f"**{top.get('max_subsidence_mm')} mm** and mean coherence loss of "
            f"**{top.get('mean_delta_coh')}**, indicating surface disruption consistent with a "
            f"landslide or rapid ground subsidence. Overall risk is **{level}** "
            f"({overall:.0f}/100). "
            + ("Immediate field verification and notification of local emergency services is "
               "recommended." if level in ("CRITICAL", "HIGH") else
               "Continued monitoring with upcoming NISAR acquisitions is recommended.")
        )
    else:
        st.success("No significant deformation events detected with the current settings.")

# --- Tab 5: Team ------------------------------------------------------------
with tab5:
    st.markdown("<h1 style='text-align:center;margin-bottom:0'>Meet the Team</h1>"
                "<p style='text-align:center;color:#aaa;font-size:18px'>"
                "The brilliant minds behind NISAR Disaster Detection Platform</p>",
                unsafe_allow_html=True)

    team = [
        ("Aruzhan Lastayeva", "Lead Researcher & ML Engineer"),
        ("Aiyazhan Lastayeva", "SAR Data Scientist"),
        ("Aisha Karimova", "Deep Learning Engineer"),
        ("Taniris Sarseken", "Geospatial Analyst"),
        ("Asset Tuzelbayev", "Backend & Visualization Engineer"),
    ]

    def member_card(name, role):
        initials = "".join(part[0] for part in name.split()[:2]).upper()
        # Kept on unindented lines so Markdown doesn't render it as a code block
        return f"""<div style="
background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
border: 1px solid rgba(255,75,75,0.3);
border-radius: 16px;
padding: 28px 20px;
text-align: center;
margin: 10px 0;
box-shadow: 0 4px 20px rgba(255,75,75,0.1);
">
<div style="
width: 72px; height: 72px;
border-radius: 50%;
background: linear-gradient(135deg, #FF4B4B, #c0392b);
margin: 0 auto 16px;
display: flex; align-items: center; justify-content: center;
font-family: 'Playfair Display', Georgia, serif !important;
font-size: 24px;
font-weight: 900 !important;
color: white;
letter-spacing: -1px;
">{initials}</div>
<h3 style="color: #ffffff; margin: 0 0 12px; font-size: 16px;">{name}</h3>
<div style="
background: rgba(255,75,75,0.1);
border-radius: 20px;
padding: 4px 12px;
display: inline-block;
font-size: 11px;
color: #aaa;
">NISAR Team · 2026</div>
</div>"""

    row1 = st.columns(3)
    for col, (name, role) in zip(row1, team[:3]):
        with col:
            with st.container():
                st.markdown(member_card(name, role), unsafe_allow_html=True)

    row2 = st.columns([1, 1, 1, 1, 1])
    for col, (name, role) in zip([row2[1], row2[3]], team[3:]):
        with col:
            with st.container():
                st.markdown(member_card(name, role), unsafe_allow_html=True)

    st.divider()
    st.markdown("## About the Project")
    st.markdown(
        "This platform was developed during a 48-hour hackathon to demonstrate the full "
        "pipeline of SAR interferometry data processing — from raw NISAR GUNW products to "
        "actionable disaster risk maps powered by deep learning.")
    st.markdown("<p style='text-align:center;font-size:18px;margin-top:30px'>"
                "Built with love for NASA Space Apps Challenge 2025</p>",
                unsafe_allow_html=True)
