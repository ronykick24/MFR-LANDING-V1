import numpy as np
import pandas as pd

def minimum_curvature(md, inc_deg, azm_deg):
    md = np.asarray(md, dtype=float)
    inc = np.deg2rad(np.asarray(inc_deg, dtype=float))
    azm = np.deg2rad(np.asarray(azm_deg, dtype=float))

    n = np.zeros_like(md)
    e = np.zeros_like(md)
    tvd = np.zeros_like(md)

    for i in range(1, len(md)):
        dmd = md[i] - md[i - 1]
        if dmd <= 0:
            dmd = 1e-6

        cos_dogleg = (
            np.sin(inc[i - 1]) * np.sin(inc[i]) * np.cos(azm[i] - azm[i - 1])
            + np.cos(inc[i - 1]) * np.cos(inc[i])
        )
        cos_dogleg = np.clip(cos_dogleg, -1.0, 1.0)
        dogleg = np.arccos(cos_dogleg)
        rf = 1.0 if dogleg < 1e-8 else 2.0 / dogleg * np.tan(dogleg / 2.0)

        tvd[i] = tvd[i - 1] + 0.5 * dmd * (np.cos(inc[i - 1]) + np.cos(inc[i])) * rf
        n[i] = n[i - 1] + 0.5 * dmd * (np.sin(inc[i - 1]) * np.cos(azm[i - 1]) + np.sin(inc[i]) * np.cos(azm[i])) * rf
        e[i] = e[i - 1] + 0.5 * dmd * (np.sin(inc[i - 1]) * np.sin(azm[i - 1]) + np.sin(inc[i]) * np.sin(azm[i])) * rf

    hd = np.sqrt(n**2 + e**2)
    return pd.DataFrame({"MD": md, "INC": np.rad2deg(inc), "AZM": np.rad2deg(azm), "TVD": tvd, "HD": hd})
