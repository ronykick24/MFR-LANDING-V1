# MFR-LANDING (Geosteering) — Streamlit

Repositorio final para geosteering con MFR no-azimutal:

- **Landing / Geostopping**: target BELOW (entrar al target resistivo)
- **In-zone**: target ROOF (distancia bajo techo)
- **Offset LAS**: GR autodetect por unidades API/GAPI y resistividades MFR (RPS2/RPM2/RPD2...)
- **Marker por click** en GR o Resistividad (ambos)
- **QC físico**: HORN / DIEL / ANI / INV con modal explicativo
- **Tabla DTBB** clara y auditable

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Cloud)
- Branch: `main`
- Main file path: `app.py`
