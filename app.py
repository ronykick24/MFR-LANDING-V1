import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import load_yaml
from src.io import load_csv
from src.trajectory import minimum_curvature
from src.inversion import invert_series
from src.geometry import dtbb_ft, lead_md_ft
from src.decision import decision_events
from src.qc.scores import horn_score, dielectric_score, anisotropy_score, invasion_score


# ============================================================
# Streamlit setup
# ============================================================
st.set_page_config(page_title="MFR-LANDING Geosteering", layout="wide")

cfg = load_yaml("config/channels.yaml")
thr = load_yaml("config/thresholds.yaml")


# ============================================================
# Session state
# ============================================================
if "run_flag" not in st.session_state:
    st.session_state.run_flag = False

# Curtain picks
if "pick_mode" not in st.session_state:
    st.session_state.pick_mode = "TOP"  # TOP o BASE
if "top_pick" not in st.session_state:
    st.session_state.top_pick = None  # {"MD":..., "HD":..., "TVD":...}
if "base_pick" not in st.session_state:
    st.session_state.base_pick = None  # {"MD":..., "HD":..., "TVD":...}
if "dip_app_auto" not in st.session_state:
    st.session_state.dip_app_auto = None
if "tvt_auto" not in st.session_state:
    st.session_state.tvt_auto = None

# MD ahead / TVD detection outputs (Landing)
if "md_ahead_top" not in st.session_state:
    st.session_state.md_ahead_top = None
if "tvd_det_top" not in st.session_state:
    st.session_state.tvd_det_top = None

# Physics dialog gating
if "auto_explain" not in st.session_state:
    st.session_state.auto_explain = True
if "last_physics" not in st.session_state:
    st.session_state.last_physics = None


# ============================================================
# Physics dialog text
# ============================================================
PHYSICS_TEXT = {
    "HORN": {
        "title": "Polarization horn (artefacto EM)",
        "what": (
            "Pico artificial por acumulación de carga/discontinuidad EM al cruzar un límite con alto contraste. "
            "Puede inflar/alterar la respuesta de resistividad aparente y simular proximidad falsa a un límite."
        ),
        "signature": [
            "Spike localizado cerca del límite",
            "Phase suele sobre-responder más que Attenuation",
            "Long spacing suele mostrar mayor overshoot",
        ],
        "why_not_dtb": [
            "Es sobre-respuesta física (overshoot), no cambio real de Rt",
            "Puede hacer que DTBB parezca muy pequeño cuando es artefacto",
        ],
        "what_app_does": [
            "Baja Confidence (gated)",
            "Suprime STOP automático y muestra HOLD/QC",
            "Sugiere corroboración con Earth Model/offset/GR",
        ],
    },
    "DIEL": {
        "title": "Dielectric effect (ε alta)",
        "what": (
            "Si ε real > ε asumida: Phase lee más bajo y Att más alto (típicamente más fuerte en 2MHz). "
            "Esto puede producir separación de curvas que no corresponde a un boundary cercano."
        ),
        "signature": [
            "Att >> Phase en mismo spacing",
            "Separación más fuerte en 2MHz que 400kHz",
        ],
        "why_not_dtb": [
            "Separación puede deberse a ε, no a un límite",
            "Att puede ser engañosa para DTBB si domina ε",
        ],
        "what_app_does": [
            "Sugiere penalizar/excluir Att",
            "Mantiene Phase como driver",
        ],
    },
    "ANI": {
        "title": "Electric anisotropy (Rh≠Rv)",
        "what": (
            "Separación suave deep–shallow correlacionada con ángulo relativo (INC/dip), sin boundary cercano. "
            "Un modelo isotrópico puede interpretarlo erróneamente como proximidad a límite."
        ),
        "signature": [
            "Separación persistente (no spike puntual)",
            "Correlación con INC/ángulo relativo",
        ],
        "why_not_dtb": [
            "ANI puede simular efecto de límite",
            "DTBB puede quedar sesgado si no se contempla Rh/Rv",
        ],
        "what_app_does": [
            "Inflar incertidumbre / bajar confidence",
            "Evitar STOP por DTBB crudo sin corroboración",
        ],
    },
    "INV": {
        "title": "Invasion / borehole / bed proximity",
        "what": (
            "Invasion y efectos de pozo pueden cambiar el orden shallow/deep y simular boundary effect. "
            "Depende del mud system y condiciones de pozo."
        ),
        "signature": [
            "Deep>Shallow o Shallow>Deep persistente",
            "Separación sostenida dependiente de condiciones",
        ],
        "why_not_dtb": [
            "Separación puede ser radial (invasion), no límite remoto",
            "DTBB puede ser falso si se asume homogeneidad ideal",
        ],
        "what_app_does": [
            "Baja confidence",
            "Sugiere corroboración con offset/GR/contexto",
        ],
    },
}


def show_physics_dialog(kind: str):
    info = PHYSICS_TEXT[kind]

    @st.dialog(info["title"], width="large")
    def _dlg():
        st.markdown(f"### {info['title']}")
        st.write(info["what"])

        st.markdown("**Firma típica (lo que estás viendo):**")
        for s in info["signature"]:
            st.write(f"- {s}")

        st.markdown("**Por qué NO es DTB/DTBB directo:**")
        for s in info["why_not_dtb"]:
            st.write(f"- {s}")

        st.markdown("**Qué hace el software:**")
        for s in info["what_app_does"]:
            st.write(f"- {s}")

        if st.button("Cerrar"):
            st.rerun()

    _dlg()


def get_selected_points(event):
    """
    Streamlit devuelve un dict-like para selecciones de Plotly.
    Diferentes versiones pueden usar:
      event["selection"]["points"] o event["select"]["points"].
    """
    if event is None:
        return []
    try:
        if "selection" in event and isinstance(event["selection"], dict):
            return event["selection"].get("points", []) or []
    except Exception:
        pass
    try:
        if "select" in event and isinstance(event["select"], dict):
            return event["select"].get("points", []) or []
    except Exception:
        pass
    return []


# ============================================================
# Picks math: dip_app auto + surfaces + intercept
# ============================================================
def compute_dip_app_deg_from_picks(top_pick, base_pick):
    # dip_app = atan(dTVD/dHD)
    dHD = float(base_pick["HD"] - top_pick["HD"])
    dTVD = float(base_pick["TVD"] - top_pick["TVD"])
    if abs(dHD) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(dTVD, dHD)))


def top_surface_tvd(hd, top_pick, dip_app_deg):
    slope = np.tan(np.radians(dip_app_deg))
    return float(top_pick["TVD"] + slope * (hd - top_pick["HD"]))


def compute_tvt_from_picks(top_pick, base_pick, dip_app_deg):
    # TVT = TVD_basePick - TVD_top_at_baseHD
    tvd_top_at_base_hd = top_surface_tvd(base_pick["HD"], top_pick, dip_app_deg)
    return float(base_pick["TVD"] - tvd_top_at_base_hd)


def base_surface_tvd(hd, top_pick, dip_app_deg, tvt):
    return float(top_surface_tvd(hd, top_pick, dip_app_deg) + tvt)


def find_intercept_md(md, hd, tvd, tvd_surface_func, md0):
    """
    Encuentra primer cruce por delante de md0 entre:
      f(MD)=TVD_well(MD)-TVD_surface(HD_well(MD)) = 0
    """
    md = np.asarray(md, dtype=float)
    hd = np.asarray(hd, dtype=float)
    tvd = np.asarray(tvd, dtype=float)

    i0 = int(np.searchsorted(md, md0))
    i0 = max(0, min(i0, len(md) - 2))

    f_prev = tvd[i0] - tvd_surface_func(hd[i0])
    for i in range(i0 + 1, len(md)):
        f = tvd[i] - tvd_surface_func(hd[i])
        if np.sign(f) == 0:
            return float(md[i]), float(hd[i]), float(tvd[i])
        if np.sign(f) != np.sign(f_prev):
            # cruce entre i-1 e i
            m1, m2 = md[i - 1], md[i]
            f1, f2 = f_prev, f
            t = 0.5 if abs(f2 - f1) < 1e-12 else (0 - f1) / (f2 - f1)
            t = float(np.clip(t, 0, 1))
            md_hit = float(m1 + t * (m2 - m1))
            hd_hit = float(hd[i - 1] + t * (hd[i] - hd[i - 1]))
            tvd_hit = float(tvd[i - 1] + t * (tvd[i] - tvd[i - 1]))
            return md_hit, hd_hit, tvd_hit
        f_prev = f

    return None


# ============================================================
# UI header + tabs
# ============================================================
st.title("MFR-LANDING Geosteering — TOP/BASE picks + MD-ahead + TVD detection")

tab_rt, tab_la, tab_land = st.tabs(
    ["Real-Time (Geosteering)", "Pseudo Look-Ahead", "Landing Calculator"]
)


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Workflow")

    phase = st.radio(
        "Mode",
        ["Landing/Geostopping (MD ahead to TOP)", "In-zone (MD ahead TOP + clearance BASE)"],
        index=0,
        help="Landing: MD ahead a TOP + TVD detection. In-zone: MD ahead a TOP y clearance a BASE."
    )
    mode = "landing" if phase.startswith("Landing") else "inzone"

    st.header("Input")
    up = st.file_uploader("Upload MFR CSV (MD, INC, RPS2..)", type=["csv"])
    up_s = st.file_uploader("Upload Survey CSV (MD, INC, AZM)", type=["csv"])

    st.header("Curtain picks (click on curtain)")
    st.session_state.pick_mode = st.radio("Pick mode", ["TOP", "BASE"], index=0 if st.session_state.pick_mode == "TOP" else 1)

    if st.button("Clear TOP/BASE picks"):
        st.session_state.top_pick = None
        st.session_state.base_pick = None
        st.session_state.dip_app_auto = None
        st.session_state.tvt_auto = None
        st.session_state.md_ahead_top = None
        st.session_state.tvd_det_top = None
        st.rerun()

    st.header("MFR geometry")
    dbb = st.number_input(
        "DBB sensor→bit [ft]",
        value=float(cfg["defaults"]["dbb_ft"]),
        step=1.0,
        help="DTBB = DTB_sensor − DBB·sin(alpha). alpha usa INC y dip_app."
    )
    bit_margin = st.number_input("Bit margin [ft]", value=float(cfg["defaults"]["bit_margin_ft"]), step=0.5)

    st.header("Thresholds (DTBB)")
    warn_ft = st.number_input("WARN (ft)", value=float(thr["dtb"]["warn_ft"]), step=0.5)
    action_ft = st.number_input("ACTION (ft)", value=float(thr["dtb"]["action_ft"]), step=0.5)
    stop_ft = st.number_input("STOP (ft)", value=float(thr["dtb"]["stop_ft"]), step=0.5)

    st.header("QC gates")
    horn_gate = st.number_input("Horn angle gate (deg)", value=float(thr["qc"]["horn_angle_gate_deg"]), step=1.0)
    horn_contrast_min = st.number_input("Horn contrast min", value=float(thr["qc"]["horn_contrast_min"]), step=0.5)
    horn_suppress = st.number_input("Suppress STOP if HORN ≥", value=float(thr["qc"]["horn_suppress_stop_ge"]), step=0.05)

    st.header("Explain physics")
    st.session_state.auto_explain = st.toggle("Auto-explain dialogs", value=st.session_state.auto_explain)

    if st.button("Run"):
        st.session_state.run_flag = True


# ============================================================
# Load data
# ============================================================
if up is None:
    df = pd.read_csv("data/sample_mfr.csv")
    st.info("Using data/sample_mfr.csv (demo)")
else:
    df = load_csv(up)

if "MD" not in df.columns:
    st.error("CSV must contain MD column")
    st.stop()

if "INC" not in df.columns:
    df["INC"] = 90.0

# Survey trajectory
traj = None
try:
    if up_s is None:
        s = pd.read_csv("data/sample_survey.csv")
    else:
        s = pd.read_csv(up_s)

    if {"MD", "INC", "AZM"}.issubset(s.columns):
        traj = minimum_curvature(s["MD"].values, s["INC"].values, s["AZM"].values)
except Exception:
    traj = None

with tab_rt:
    st.subheader("Preview")
    st.dataframe(df.head(12), use_container_width=True)

if not st.session_state.run_flag:
    st.stop()


# ============================================================
# Inversion MFR (DTB)
# ============================================================
channels = cfg["mfr_channels"]

# Dip usado para proyección MFR: si hay dip_auto (picks), usarlo; si no, fallback a default.
dip_for_mfr = st.session_state.dip_app_auto if st.session_state.dip_app_auto is not None else float(cfg["defaults"]["dip_app_deg"])

out = invert_series(df, channels, dbb_ft=dbb, dip_app_deg=dip_for_mfr, mode=("below" if mode == "landing" else "roof"))

df2 = df.copy()
for col in out.columns:
    df2[col] = out[col].values

# QC scores
df2["HORN"] = horn_score(df2, df2["INC"].values, dip_for_mfr, horn_angle_gate_deg=horn_gate, contrast_min=horn_contrast_min)
df2["DIEL"] = dielectric_score(df2)
df2["ANI"] = anisotropy_score(df2, df2["INC"].values)
df2["INV"], df2["INV_PATTERN"] = invasion_score(df2, mud_system=("WBM"))  # puedes parametrizar si quieres

df2["Confidence_gated"] = df2["Confidence"].values * (1.0 - np.clip(df2["HORN"].values, 0.0, 1.0))

# Events
events = decision_events(out, warn_ft, action_ft, stop_ft)

# Last point
last = df2.dropna(subset=["DTB_Target_Bit_ft"]).tail(1)
inc_last = float(last["INC"].values[0])
dtb_sensor_last = float(last["DTB_Target_Sensor_ft"].values[0])
dtbb_last = dtbb_ft(dtb_sensor_last, dbb, inc_last, dip_for_mfr)
conf_last = float(last["Confidence_gated"].values[0])

horn_last = float(last["HORN"].values[0])
diel_last = float(last["DIEL"].values[0])
ani_last = float(last["ANI"].values[0])
inv_last = float(last["INV"].values[0])

# Phenomenon selection
phen = None
if horn_last >= float(thr["qc"]["horn_suppress_stop_ge"]):
    phen = "HORN"
elif diel_last >= float(thr["qc"]["diel_penalize_att_ge"]):
    phen = "DIEL"
elif ani_last >= float(thr["qc"]["ani_inflate_unc_ge"]):
    phen = "ANI"
elif inv_last >= float(thr["qc"]["inv_lower_conf_ge"]):
    phen = "INV"

in_decision_zone = (dtbb_last <= warn_ft)

if st.session_state.auto_explain and in_decision_zone and phen and st.session_state.last_physics != phen:
    st.session_state.last_physics = phen
    show_physics_dialog(phen)


# ============================================================
# Real-Time UI
# ============================================================
with tab_rt:
    col_tracks, col_curtain, col_dec = st.columns([1.25, 1.70, 1.05], gap="large")

    # ------------------ TRACKS ------------------
    with col_tracks:
        st.subheader("MFR Tracks (training-style)")
        md = df2["MD"].values

        fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                            column_widths=[0.40, 0.30, 0.30])

        # Phase 2MHz (RPD2 driver)
        for name, color in [("RPS2", "#666666"), ("RPM2", "#1f77b4"), ("RPD2", "#d62728")]:
            if name in df2.columns:
                fig.add_trace(go.Scatter(x=df2[name], y=md, name=name, mode="lines", line=dict(color=color)), 1, 1)
        fig.update_xaxes(type="log", title_text="Phase 2MHz", row=1, col=1)

        # QC
        fig.add_trace(go.Scatter(x=df2["HORN"], y=md, name="HORN", mode="lines", line=dict(color="red")), 1, 2)
        fig.add_trace(go.Scatter(x=df2["DIEL"], y=md, name="DIEL", mode="lines", line=dict(color="purple")), 1, 2)
        fig.add_trace(go.Scatter(x=df2["ANI"], y=md, name="ANI", mode="lines", line=dict(color="green")), 1, 2)
        fig.add_trace(go.Scatter(x=df2["INV"], y=md, name="INV", mode="lines", line=dict(color="orange")), 1, 2)
        fig.update_xaxes(title_text="QC (0-1)", row=1, col=2)

        # Outputs
        fig.add_trace(go.Scatter(x=df2["DTB_Target_Sensor_ft"], y=md, name="DTB_sensor", mode="lines", line=dict(color="blue")), 1, 3)
        fig.add_trace(go.Scatter(x=df2["DTB_Target_Bit_ft"], y=md, name="DTB_bit", mode="lines", line=dict(color="red")), 1, 3)
        fig.add_trace(go.Scatter(x=df2["Confidence_gated"], y=md, name="Conf_gated", mode="lines", line=dict(color="black", dash="dash")), 1, 3)
        fig.update_xaxes(title_text="DTB/Conf", row=1, col=3)

        fig.update_yaxes(autorange="reversed", title_text="MD", row=1, col=1)
        fig.update_layout(height=650, legend=dict(orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

    # ------------------ CURTAIN (HD vs TVD) ------------------
    with col_curtain:
        st.subheader("Curtain (HD vs TVD) — click wellpath to pick TOP/BASE")

        if traj is None:
            st.info("Upload Survey CSV to enable Curtain + MD-ahead/TVD detection.")
        else:
            md_arr = traj["MD"].values
            hd_arr = traj["HD"].values
            tvd_arr = traj["TVD"].values

            md_current = float(df2["MD"].iloc[-1])
            hd_current = float(np.interp(md_current, md_arr, hd_arr))
            tvd_current = float(np.interp(md_current, md_arr, tvd_arr))

            # If TOP & BASE exist, compute dip + TVT
            if st.session_state.top_pick and st.session_state.base_pick:
                dip_auto = compute_dip_app_deg_from_picks(st.session_state.top_pick, st.session_state.base_pick)
                if dip_auto is not None:
                    st.session_state.dip_app_auto = dip_auto
                    st.session_state.tvt_auto = compute_tvt_from_picks(st.session_state.top_pick, st.session_state.base_pick, dip_auto)

            fig_c = go.Figure()

            # wellpath selectable (lines+markers is key for click selection)
            fig_c.add_trace(go.Scatter(
                x=hd_arr, y=tvd_arr,
                mode="lines+markers",
                marker=dict(size=3, color="black"),
                line=dict(color="black", width=3),
                name="Wellpath"
            ))

            # Draw TOP/BASE + band if defined
            if st.session_state.top_pick and st.session_state.dip_app_auto is not None and st.session_state.tvt_auto is not None:
                dip_auto = st.session_state.dip_app_auto
                tvt = st.session_state.tvt_auto

                hd_line = np.linspace(float(np.min(hd_arr)), float(np.max(hd_arr)), 250)
                top_line = [top_surface_tvd(x, st.session_state.top_pick, dip_auto) for x in hd_line]
                base_line = [base_surface_tvd(x, st.session_state.top_pick, dip_auto, tvt) for x in hd_line]

                # Target band
                fig_c.add_trace(go.Scatter(
                    x=np.concatenate([hd_line, hd_line[::-1]]),
                    y=np.concatenate([np.array(top_line), np.array(base_line)[::-1]]),
                    fill="toself",
                    fillcolor="rgba(255,255,0,0.20)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Target band"
                ))

                fig_c.add_trace(go.Scatter(x=hd_line, y=top_line, mode="lines+markers",
                                           marker=dict(size=2, color="blue"),
                                           line=dict(color="blue", dash="dash"), name="TOP"))
                fig_c.add_trace(go.Scatter(x=hd_line, y=base_line, mode="lines+markers",
                                           marker=dict(size=2, color="blue"),
                                           line=dict(color="blue", dash="dash"), name="BASE"))

                # Landing/Geostopping: MD ahead to TOP + TVD detection
                def tvd_top_func(hd_val):
                    return top_surface_tvd(hd_val, st.session_state.top_pick, dip_auto)

                hit = find_intercept_md(md_arr, hd_arr, tvd_arr, tvd_top_func, md_current)
                if hit is not None:
                    md_hit, hd_hit, tvd_hit = hit
                    tvd_top_hit = tvd_top_func(hd_hit)

                    md_ahead_top = md_hit - md_current
                    tvd_det = tvd_top_hit - tvd_current  # + si TOP está por debajo (antes de tocar)

                    st.session_state.md_ahead_top = float(md_ahead_top)
                    st.session_state.tvd_det_top = float(tvd_det)

                    # Callouts (como tu imagen)
                    fig_c.add_annotation(
                        x=hd_hit, y=tvd_top_hit,
                        text=f"{md_ahead_top:.0f} ft MD ahead (TOP)",
                        showarrow=True, arrowhead=2, ax=-60, ay=-40,
                        font=dict(color="blue")
                    )
                    fig_c.add_annotation(
                        x=hd_hit, y=(tvd_top_hit + tvd_current) / 2,
                        text=f"{tvd_det:.0f} ft TVD detection",
                        showarrow=True, arrowhead=2, ax=60, ay=0,
                        font=dict(color="blue")
                    )

                # In-zone: clearance to BASE
                if mode == "inzone":
                    tvd_base_here = base_surface_tvd(hd_current, st.session_state.top_pick, dip_auto, tvt)
                    clearance_base_tvd = tvd_base_here - tvd_current
                    st.caption(f"In-zone: Clearance to BASE (TVD) = {clearance_base_tvd:.1f} ft")

            # Show picks
            for label, pick in [("TOP pick", st.session_state.top_pick), ("BASE pick", st.session_state.base_pick)]:
                if pick:
                    fig_c.add_trace(go.Scatter(
                        x=[pick["HD"]], y=[pick["TVD"]],
                        mode="markers",
                        marker=dict(size=12, color="black", symbol="x"),
                        name=label
                    ))

            fig_c.update_yaxes(autorange="reversed", title="TVD")
            fig_c.update_xaxes(title="HD")
            fig_c.update_layout(height=650, legend=dict(orientation="h"), clickmode="event+select")

            ev = st.plotly_chart(
                fig_c,
                key="curtain_chart",
                on_select="rerun",
                selection_mode=("points",),
                use_container_width=True
            )

            pts = get_selected_points(ev)
            if pts:
                x = pts[0].get("x", None)
                y = pts[0].get("y", None)
                if x is not None and y is not None:
                    # Pick anchored on clicked wellpath point
                    # Convert clicked (HD,TVD) into MD by nearest match on trajectory
                    # Find nearest point in (HD,TVD) along wellpath
                    dist = (hd_arr - float(x))**2 + (tvd_arr - float(y))**2
                    j = int(np.argmin(dist))
                    md_pick = float(md_arr[j])

                    pick = {"MD": md_pick, "HD": float(hd_arr[j]), "TVD": float(tvd_arr[j])}

                    if st.session_state.pick_mode == "TOP":
                        st.session_state.top_pick = pick
                        st.success(f"TOP pick set: MD={pick['MD']:.1f}, HD={pick['HD']:.1f}, TVD={pick['TVD']:.1f}")
                    else:
                        st.session_state.base_pick = pick
                        st.success(f"BASE pick set: MD={pick['MD']:.1f}, HD={pick['HD']:.1f}, TVD={pick['TVD']:.1f}")
                    st.rerun()

    # ------------------ DECISION PANEL ------------------
    with col_dec:
        st.subheader("Decision + Distances")

        st.metric("DTBB bit (ft)", f"{dtbb_last:.2f}")
        st.metric("Confidence (gated)", f"{conf_last:.2f}")

        if horn_last >= horn_suppress:
            st.error("HOLD (QC: Polarization horn)")
        elif dtbb_last <= stop_ft:
            st.error("STOP")
        elif dtbb_last <= action_ft:
            st.warning("ACTION")
        elif dtbb_last <= warn_ft:
            st.info("WARN")
        else:
            st.success("OK")

        st.subheader("Landing/Geostopping")
        if mode == "landing":
            if st.session_state.md_ahead_top is not None and st.session_state.tvd_det_top is not None:
                st.metric("MD ahead to TOP (ft)", f"{st.session_state.md_ahead_top:.1f}")
                st.metric("TVD detection to TOP (ft)", f"{st.session_state.tvd_det_top:.1f}")
            else:
                st.info("Pick TOP and BASE on Curtain to compute MD-ahead + TVD detection.")

        st.subheader("In-zone")
        if mode == "inzone":
            st.info("Clearance to BASE is shown under the Curtain caption (requires TOP+BASE picks).")

        st.subheader("DTBB audit table (clear)")
        alpha_deg = abs(inc_last - (90.0 - dip_for_mfr))
        d_sb = dbb * np.sin(np.deg2rad(alpha_deg))
        landing_table = pd.DataFrame({
            "Parameter": [
                "INC (deg)",
                "dip_app used (deg)",
                "alpha = |INC − (90 − dip)| (deg)",
                "DBB (ft)",
                "d = DBB·sin(alpha) (ft)",
                "DTB sensor (ft)",
                "DTBB (ft)"
            ],
            "Value": [
                f"{inc_last:.2f}",
                f"{dip_for_mfr:.2f}",
                f"{alpha_deg:.2f}",
                f"{dbb:.1f}",
                f"{d_sb:.2f}",
                f"{dtb_sensor_last:.2f}",
                f"{dtbb_last:.2f}"
            ]
        })
        st.table(landing_table)

        st.subheader("Events")
        st.dataframe(events, use_container_width=True)

        st.download_button(
            "Download results_qc.csv",
            data=df2.to_csv(index=False).encode("utf-8"),
            file_name="results_qc.csv",
            mime="text/csv"
        )


# ============================================================
# Pseudo Look-Ahead
# ============================================================
with tab_la:
    st.markdown("## Pseudo Look-Ahead")
    st.caption("LA = (DOD / tan(alpha)) − DBB  (geométrico tipo training)")

    dod = st.number_input("DOD (ft)", value=10.0, step=0.5)

    dip_used = st.session_state.dip_app_auto if st.session_state.dip_app_auto is not None else dip_for_mfr
    alpha = abs(inc_last - (90.0 - dip_used))
    t = np.tan(np.deg2rad(alpha))
    la = 0.0 if abs(t) < 1e-9 else max(dod / t - dbb, 0.0)

    st.metric("alpha (deg)", f"{alpha:.2f}")
    st.metric("Pseudo LA (ft)", f"{la:.1f}")


# ============================================================
# Landing calculator
# ============================================================
with tab_land:
    st.markdown("## Landing Calculator")
    st.caption("R = 5730/B; ΔTVD = R − R·cos(Δθ)")

    B = st.number_input("Build rate B (deg/100ft)", value=3.0, step=0.1)
    inc0 = st.number_input("INC actual (deg)", value=87.0, step=0.1)
    inc1 = st.number_input("INC objetivo (deg)", value=90.0, step=0.1)
    above = st.number_input("Sobre Top (ft)", value=2.0, step=0.5)

    R = 5730.0 / max(B, 1e-6)
    dtheta = abs(inc1 - inc0)
    dTVD = R - R * np.cos(np.deg2rad(dtheta))
    outcome = dTVD - above

    st.metric("R (ft)", f"{R:.0f}")
    st.metric("ΔTVD (ft)", f"{dTVD:.2f}")
    st.metric("Outcome vs Top (ft)", f"{outcome:+.2f}")
