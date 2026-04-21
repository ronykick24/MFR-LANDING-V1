# MFR-LANDING (Geosteering) — Streamlit

## Qué hace
- Landing/Geostopping: **MD ahead a TOP** + **TVD detection** antes de tocar el top.
- In-zone: **MD ahead a TOP** + **clearance a BASE (TVD)** para evitar salir por abajo.
- Picks **TOP/BASE por click** en Curtain (HD vs TVD) y **dip_app automático**.
- Banda objetivo **entre TOP y BASE** sombreada.
- QC físico (HORN/DIEL/ANI/INV) con modal explicativo (no confundir con DTBB).
- MFR no-azimutal: el lado del límite lo define el modo (Landing vs In-zone).

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
