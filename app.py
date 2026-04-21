import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import load_yaml
from src.io import load_csv
from src.io_las import load_las
from src.mnemonics import find_gr_curve, is_resistivity_curve, classify_mfr_mnemonic
from src.trajectory import minimum_curvature
from src.inversion import invert_series
from src.geometry import d_sensor_to_bit_ft, lead_md_ft, dtbb_ft
from src.decision import decision_events
from src.qc.scores import horn_score, dielectric_score, anisotropy_score, invasion_score

st.set_page_config(page_title="MFR Geosteering", layout="wide")

cfg = load_yaml("config/channels.yaml")
thr = load_yaml("config/thresholds.yaml")

# --- Session state ---
if "run_flag" not in st.session_state:
    st.session_state.run_flag = False
if "marker_md" not in st.session_state:
    st.session_state.marker_md = None
if "last_physics" not in st.session_state:
    st.session_state.last_physics = None
if "auto_explain" not in st.session_state:
    st.session_state.auto_explain = True

# --- Physics text ---
PHYSICS_TEXT = {
  "HORN": {
    "title": "Polarization horn (artefacto EM)",
    "what": "Pico artificial por acumulación de carga/discontinuidad EM al cruzar límite con alto contraste.",
    "signature": [
      "Spike localizado cerca del límite",
      "Phase suele sobre-responder más que Attenuation",
      "Long spacing suele mostrar mayor overshoot",
    ],
    "why_not_dtb": [
      "Es sobre-respuesta física (overshoot), no cambio real de Rt",
      "Puede hacer que DTBB parezca pequeño cuando es artefacto",
    ],
    "what_app_does": [
      "Baja Confidence (gated)",
      "Suprime STOP automático y muestra HOLD/QC",
      "Sugiere corroboración con offset/GR/contexto",
    ],
  },
  "DIEL": {
    "title": "Dielectric effect (ε alta)",
    "what": "ε real>ε asumida: Phase lee más bajo, Att más alto; efecto mayor en 2MHz que 400kHz.",
    "signature": [
      "Att >> Phase en mismo spacing",
      "Separación más fuerte en 2MHz que en 400kHz",
    ],
    "why_not_dtb": [
      "La separación puede venir de ε, no de límite",
      "Att puede ser engañosa para DTBB si domina ε",
    ],
    "what_app_does": [
      "Penaliza/excluye Att en interpretación",
      "Mantiene Phase como driver",
    ],
  },
  "ANI": {
    "title": "Electric anisotropy (Rh≠Rv)",
    "what": "Separación suave deep–shallow con ángulo relativo (INC/dip), sin boundary cercano.",
    "signature": [
      "Separación persistente (no spike puntual)",
      "Correlación con INC/ángulo relativo",
    ],
    "why_not_dtb": [
      "Modelo isotrópico puede interpretar ANI como límite",
      "DTBB puede quedar sesgado si no se contempla Rh/Rv",
    ],
    "what_app_does": [
      "Inflar incertidumbre / bajar confidence",
      "Evitar STOP por DTBB crudo sin corroboración",
    ],
  },
  "INV": {
    "title": "Invasion / borehole / bed proximity",
    "what": "Invasion/pozo cambia orden shallow/deep y puede simular boundary effect.",
    "signature": [
      "Deep>Shallow o Shallow>Deep persistente",
      "Patrón dependiente de mud system",
    ],
    "why_not_dtb": [
      "Separación puede ser radial (invasion), no límite remoto",
      "DTBB puede ser falso si se asume homogeneidad",
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


st.title("MFR Geosteering (No-Azimuthal) — Landing/Geostopping + In-zone roof distance")

# --- Tabs ---
tab_rt, tab_la, tab_pre, tab_land = st.tabs([
    "Real-Time (Geosteering)",
    "Pseudo Look-Ahead",
    "Pre-Job Modelling",
    "Landing Calculator"
])

# --- Sidebar ---
with st.sidebar:
    st.header("Workflow")
    phase = st.radio(
        "Phase",
        ["Landing: target BELOW (enter resistive target)", "In-zone: target ROOF (distance under roof)"],
        index=0,
        help="Define el lado operativo del límite: Landing→Below, In-zone→Roof."
    )
    mode = "below" if phase.startswith("Landing") else "roof"

    st.header("Input data")
    up = st.file_uploader("Upload MFR CSV (MD, INC, RPS2..)", type=["csv"], help="CSV con MD/INC y curvas MFR.")
    up_s = st.file_uploader("Upload Survey CSV (MD, INC, AZM) [optional]", type=["csv"], help="Survey para curtain (HD vs TVD).")

    st.header("Offset (Reference well)")
    off_file = st.file_uploader("Upload OFFSET LAS", type=["las","LAS"], help="LAS de pozo offset (GR + resistividades MFR).")

    st.header("Structure / Geometry")
    dip_app = st.number_input(
        "Dip aparente [deg] (down +)",
        value=float(cfg["defaults"]["dip_app_deg"]),
        step=0.5,
        help="Usado para α = |INC − (90 − dip_app)|."
    )
    dbb = st.number_input(
        "DBB sensor→bit [ft]",
        value=float(cfg["defaults"]["dbb_ft"]),
        step=1.0,
        help="Usado en d = DBB·sin(α) y DTBB = DTB_sensor − d."
    )
    bit_margin = st.number_input(
        "Bit margin [ft]",
        value=float(cfg["defaults"]["bit_margin_ft"]),
        step=0.5,
        help="Margen de seguridad al bit para LeadMD."
    )
    mud_system = st.selectbox("Mud system", ["WBM","OBM"], index=0, help="Afecta INV.")

    st.header("Thresholds (DTBB at bit)")
    warn_ft = st.number_input("WARN", value=float(thr["dtb"]["warn_ft"]), step=0.5)
    action_ft = st.number_input("ACTION", value=float(thr["dtb"]["action_ft"]), step=0.5)
    stop_ft = st.number_input("STOP", value=float(thr["dtb"]["stop_ft"]), step=0.5)

    st.header("QC gates")
    horn_gate = st.number_input("Horn angle gate [deg]", value=float(thr["qc"]["horn_angle_gate_deg"]), step=1.0)
    horn_contrast_min = st.number_input("Horn contrast min", value=float(thr["qc"]["horn_contrast_min"]), step=0.5)
    horn_suppress = st.number_input("Suppress STOP if HORN ≥", value=float(thr["qc"]["horn_suppress_stop_ge"]), step=0.05)

    st.header("Explain physics")
    st.session_state.auto_explain = st.toggle("Auto-explain (dialogs)", value=st.session_state.auto_explain)

    st.header("Zones")
    enable_zones = st.toggle("Enable optimum/hazard zones", value=True)
    opt_half = st.number_input("Optimum half-window (ft)", value=10.0, step=1.0)
    haz_half = st.number_input("Hazard half-window (ft)", value=20.0, step=1.0)

    if st.button("Run"):
        st.session_state.run_flag = True

# --- Load current well data ---
if up is None:
    df = pd.read_csv("data/sample_mfr.csv")
    st.info("Using built-in data/sample_mfr.csv")
else:
    df = load_csv(up)

if "MD" not in df.columns:
    st.error("CSV must contain MD column")
    st.stop()

if "INC" not in df.columns:
    df["INC"] = 90.0

# --- Load survey ---
traj = None
try:
    if up_s is None:
        s = pd.read_csv("data/sample_survey.csv")
    else:
        s = pd.read_csv(up_s)
    if {"MD","INC","AZM"}.issubset(s.columns):
        traj = minimum_curvature(s["MD"].values, s["INC"].values, s["AZM"].values)
except Exception:
    traj = None

with tab_rt:
    st.subheader("Preview")
    st.dataframe(df.head(12), use_container_width=True)

if not st.session_state.run_flag:
    st.stop()

# --- Inversion ---
channels = cfg["mfr_channels"]
out = invert_series(df, channels, dbb_ft=dbb, dip_app_deg=dip_app, mode=mode)

# Merge
for col in out.columns:
    df[col] = out[col].values

df2 = df

# QC
horn = horn_score(df2, df2["INC"].values, dip_app, horn_angle_gate_deg=horn_gate, contrast_min=horn_contrast_min)
diel = dielectric_score(df2)
ani = anisotropy_score(df2, df2["INC"].values)
inv, inv_pat = invasion_score(df2, mud_system=mud_system)

df2["HORN"] = horn
df2["DIEL"] = diel
df2["ANI"] = ani
df2["INV"] = inv
df2["INV_PATTERN"] = inv_pat

df2["Confidence_gated"] = df2["Confidence"].values * (1.0 - np.clip(df2["HORN"].values, 0.0, 1.0))

# Last
last = df2.dropna(subset=["DTB_Target_Bit_ft"]).tail(1)
inc_last = float(last["INC"].values[0])
dtb_s = float(last["DTB_Target_Sensor_ft"].values[0])
dtbb_last = dtbb_ft(dtb_s, dbb, inc_last, dip_app)
conf_last = float(last["Confidence_gated"].values[0])
lead_last = lead_md_ft(dtb_s, dbb, inc_last, dip_app, bit_margin_ft=bit_margin)

horn_last = float(last["HORN"].values[0])
diel_last = float(last["DIEL"].values[0])
ani_last  = float(last["ANI"].values[0])
inv_last  = float(last["INV"].values[0])

# Phenomenon
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

# --- REAL-TIME UI ---
with tab_rt:
    col_tracks, col_curtain, col_ref = st.columns([1.25, 1.5, 1.15], gap="large")

    with col_tracks:
        st.subheader("MFR Tracks (training-style)")
        md = df2["MD"].values
        fig = make_subplots(rows=1, cols=5, shared_yaxes=True, horizontal_spacing=0.03,
                            column_widths=[0.22,0.22,0.18,0.18,0.20])

        for name, color in [("RPS2","#333333"),("RPM2","#1f77b4"),("RPD2","#d62728")]:
            if name in df2.columns:
                fig.add_trace(go.Scatter(x=df2[name], y=md, name=name, mode='lines', line=dict(color=color)), 1, 1)
        fig.update_xaxes(type='log', title_text='Phase 2MHz', row=1, col=1)

        for name, color in [("RPS4","#555555"),("RPM4","#2ca02c"),("RPD4","#ff7f0e")]:
            if name in df2.columns:
                fig.add_trace(go.Scatter(x=df2[name], y=md, name=name, mode='lines', line=dict(color=color)), 1, 2)
        fig.update_xaxes(type='log', title_text='Phase 400kHz', row=1, col=2)

        for name, color in [("RAD2","#9467bd"),("RAD4","#8c564b")]:
            if name in df2.columns:
                fig.add_trace(go.Scatter(x=df2[name], y=md, name=name, mode='lines', line=dict(color=color, dash='dot')), 1, 3)
        fig.update_xaxes(type='log', title_text='Att', row=1, col=3)

        fig.add_trace(go.Scatter(x=df2['HORN'], y=md, name='HORN', mode='lines', line=dict(color='red')), 1, 4)
        fig.add_trace(go.Scatter(x=df2['DIEL'], y=md, name='DIEL', mode='lines', line=dict(color='purple')), 1, 4)
        fig.add_trace(go.Scatter(x=df2['ANI'], y=md, name='ANI', mode='lines', line=dict(color='green')), 1, 4)
        fig.add_trace(go.Scatter(x=df2['INV'], y=md, name='INV', mode='lines', line=dict(color='orange')), 1, 4)
        fig.update_xaxes(title_text='QC (0-1)', row=1, col=4)

        fig.add_trace(go.Scatter(x=df2['DTB_Target_Sensor_ft'], y=md, name='DTB_sensor', mode='lines', line=dict(color='blue')), 1, 5)
        fig.add_trace(go.Scatter(x=df2['DTB_Target_Bit_ft'], y=md, name='DTB_bit', mode='lines', line=dict(color='red')), 1, 5)
        fig.add_trace(go.Scatter(x=df2['Confidence_gated'], y=md, name='Conf_gated', mode='lines', line=dict(color='black', dash='dash')), 1, 5)
        fig.update_xaxes(title_text='DTB/Conf', row=1, col=5)

        fig.update_yaxes(autorange='reversed', title_text='MD', row=1, col=1)
        fig.update_layout(height=650, legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

    with col_curtain:
        st.subheader("Curtain (HD vs TVD)")
        if traj is None:
            st.info("Upload survey to enable curtain")
        else:
            md_log = df2['MD'].astype(float).values
            tvd = np.interp(md_log, traj['MD'].values, traj['TVD'].values)
            hd = np.interp(md_log, traj['MD'].values, traj['HD'].values)
            roof0 = float(tvd[0]); hd0 = float(hd[0])
            roof = roof0 + np.tan(np.deg2rad(dip_app))*(hd-hd0)

            f = go.Figure()
            f.add_trace(go.Scatter(x=hd, y=tvd, name='Wellpath', mode='lines', line=dict(color='black', width=3)))
            f.add_trace(go.Scatter(x=hd, y=roof, name='Roof', mode='lines', line=dict(color='brown', dash='dash')))

            if st.session_state.marker_md is not None:
                m = float(st.session_state.marker_md)
                tvd_m = float(np.interp(m, traj['MD'].values, traj['TVD'].values))
                f.add_hline(y=tvd_m, line_width=2, line_dash='dot', line_color='black')
                if enable_zones:
                    f.add_hrect(y0=tvd_m-haz_half, y1=tvd_m+haz_half, fillcolor='rgba(255,0,0,0.10)', line_width=0, layer='below')
                    f.add_hrect(y0=tvd_m-opt_half, y1=tvd_m+opt_half, fillcolor='rgba(255,215,0,0.18)', line_width=0, layer='below')

            f.update_yaxes(autorange='reversed', title='TVD')
            f.update_xaxes(title='HD')
            f.update_layout(height=650, legend=dict(orientation='h'))
            st.plotly_chart(f, use_container_width=True)

    with col_ref:
        st.subheader("Reference Well (Offset LAS) + Decision")

        # Decision
        st.metric('DTBB bit (ft)', f'{dtbb_last:.2f}')
        st.metric('Confidence (gated)', f'{conf_last:.2f}')
        st.metric('LeadMD to margin (ft)', f'{lead_last:.1f}')

        if horn_last >= horn_suppress:
            st.error('HOLD (QC: Polarization horn)')
        elif dtbb_last <= stop_ft:
            st.error('STOP')
        elif dtbb_last <= action_ft:
            st.warning('ACTION')
        elif dtbb_last <= warn_ft:
            st.info('WARN')
        else:
            st.success('OK')

        # Landing table
        alpha_deg = abs(inc_last - (90.0 - dip_app))
        d_sb = dbb * np.sin(np.deg2rad(alpha_deg))
        landing_table = pd.DataFrame({
            'Parameter': ['INC (deg)','Dip_app (deg)','Alpha (deg)','DBB (ft)','d=DBB*sin(alpha) (ft)','DTB sensor (ft)','DTBB (ft)'],
            'Value': [f'{inc_last:.2f}', f'{dip_app:.2f}', f'{alpha_deg:.2f}', f'{dbb:.1f}', f'{d_sb:.2f}', f'{dtb_s:.2f}', f'{dtbb_last:.2f}']
        })
        st.table(landing_table)

        # Offset LAS
        if off_file is None:
            st.info('Upload OFFSET LAS to enable composite')
        else:
            las = load_las(off_file)
            odf = las.df().reset_index()
            odf.rename(columns={odf.columns[0]:'DEPTH'}, inplace=True)
            md_off = odf['DEPTH'].astype(float).values

            gr_mnem = find_gr_curve(las)
            res_candidates = []
            for c in las.curves:
                unit = getattr(c,'unit','')
                mn = c.mnemonic
                ds = getattr(c,'descr','')
                if is_resistivity_curve(unit, mn, ds):
                    meta = classify_mfr_mnemonic(mn)
                    res_candidates.append({'mnemonic':mn,'meta':meta})

            if res_candidates:
                res_names = [r['mnemonic'] for r in res_candidates if r['mnemonic'] in odf.columns]
                if not res_names:
                    st.warning('No resistivity columns found in LAS dataframe')
                else:
                    default_res = 'RPD2' if 'RPD2' in res_names else res_names[0]
                    sel_res = st.selectbox('Offset resistivity', res_names, index=res_names.index(default_res))
                    meta_sel = next(r['meta'] for r in res_candidates if r['mnemonic']==sel_res)
                    st.caption(f"Detected: kind={meta_sel['kind']} | freq={meta_sel['freq']} | spacing={meta_sel['spacing']}")

                    fig_ref = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                                            subplot_titles=('GR', sel_res))
                    if gr_mnem and gr_mnem in odf.columns:
                        fig_ref.add_trace(go.Scatter(x=odf[gr_mnem], y=md_off, mode='lines+markers', marker=dict(size=3,color='green'), line=dict(color='green')), 1, 1)
                    else:
                        fig_ref.add_trace(go.Scatter(x=[0,0], y=[md_off.min(), md_off.max()]), 1, 1)

                    fig_ref.add_trace(go.Scatter(x=odf[sel_res], y=md_off, mode='lines+markers', marker=dict(size=3,color='red'), line=dict(color='red')), 1, 2)

                    if st.session_state.marker_md is not None:
                        m = float(st.session_state.marker_md)
                        fig_ref.add_hline(y=m, line_width=2, line_dash='dash', line_color='black')
                        if enable_zones:
                            fig_ref.add_hrect(y0=m-haz_half, y1=m+haz_half, fillcolor='rgba(255,0,0,0.10)', line_width=0, layer='below')
                            fig_ref.add_hrect(y0=m-opt_half, y1=m+opt_half, fillcolor='rgba(255,215,0,0.18)', line_width=0, layer='below')

                    fig_ref.update_yaxes(autorange='reversed', title_text='DEPTH')
                    fig_ref.update_xaxes(title_text='GR', row=1, col=1)
                    fig_ref.update_xaxes(type='log', title_text='Res', row=1, col=2)
                    fig_ref.update_layout(height=420, showlegend=False, clickmode='event+select')

                    ev = st.plotly_chart(fig_ref, key='ref_chart', on_select='rerun', selection_mode=('points',), use_container_width=True)
                    pts = get_selected_points(ev)
                    if pts:
                        y_clicked = pts[0].get('y', None)
                        if y_clicked is not None:
                            st.session_state.marker_md = float(y_clicked)
                            st.success(f"Marker set: MD={st.session_state.marker_md:.2f}")

        st.download_button('Download results_qc.csv', data=df2.to_csv(index=False).encode('utf-8'), file_name='results_qc.csv', mime='text/csv')

with tab_la:
    st.markdown('## Pseudo Look-Ahead')
    st.caption('LA = (DOD / tan(alpha)) − DBB')
    dod = st.number_input('DOD (ft)', value=10.0, step=0.5)
    alpha = abs(inc_last - (90.0 - dip_app))
    t = np.tan(np.deg2rad(alpha))
    la = 0.0 if abs(t) < 1e-9 else max(dod/t - dbb, 0.0)
    st.metric('alpha (deg)', f'{alpha:.2f}')
    st.metric('Pseudo LA (ft)', f'{la:.1f}')

with tab_pre:
    st.markdown('## Pre-Job Modelling (placeholder)')
    st.write('Puedes expandir a casos × trayectorias × processing (como tu slide).')

with tab_land:
    st.markdown('## Landing Calculator')
    st.caption('R = 5730/B; ΔTVD = R − R·cos(Δθ)')
    B = st.number_input('Build rate B (deg/100ft)', value=3.0, step=0.1)
    inc0 = st.number_input('INC actual (deg)', value=87.0, step=0.1)
    inc1 = st.number_input('INC objetivo (deg)', value=90.0, step=0.1)
    above = st.number_input('Sobre Top (ft)', value=2.0, step=0.5)
    R = 5730.0/max(B,1e-6)
    dtheta = abs(inc1-inc0)
    dTVD = R - R*np.cos(np.deg2rad(dtheta))
    outcome = dTVD - above
    st.metric('R (ft)', f'{R:.0f}')
    st.metric('ΔTVD (ft)', f'{dTVD:.2f}')
    st.metric('Outcome vs Top (ft)', f'{outcome:+.2f}')
