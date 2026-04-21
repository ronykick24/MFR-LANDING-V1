import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .forward import predict_channel
from .geometry import dtbb_ft

def solve_point(y_obs, channels, x0):
    def fun(x):
        R0, Rsh, h = x
        res = []
        for obs, ch in zip(y_obs, channels):
            pred = predict_channel(R0, Rsh, h, ch["h0_ft"], ch["s_ft"])
            sigma = max(ch["sigma_rel"] * max(obs, 1e-6), 1e-6)
            res.append((obs - pred) / sigma)
        return np.asarray(res, dtype=float)

    bounds = ([1e-4, 1e-4, 0.0], [1e6, 1e6, 200.0])
    sol = least_squares(fun, x0=np.asarray(x0, dtype=float), bounds=bounds, loss="huber", f_scale=1.0, max_nfev=80)
    return sol.x, sol.cost

def invert_series(df: pd.DataFrame, channels, dbb_ft: float, dip_app_deg: float, mode="below"):
    used = [ch for ch in channels if ch["name"] in df.columns and ch.get("enabled", True)]
    if len(used) < 2:
        raise ValueError("Need at least 2 enabled channels present in data")

    md = df["MD"].astype(float).values
    inc = df["INC"].astype(float).values if "INC" in df.columns else np.full_like(md, 90.0)

    def guess_R0(row):
        prefs = ["RPD2", "RPM2", "RPS2", "RPD4", "RPM4", "RPS4"]
        vals = [float(row[p]) for p in prefs if p in df.columns and np.isfinite(row[p])]
        return float(np.median(vals)) if vals else float(np.nanmedian([row[ch["name"]] for ch in used]))

    out_rows = []
    state_prev = None

    for i in range(len(df)):
        row = df.iloc[i]
        y = []
        ok = True
        for ch in used:
            v = row[ch["name"]]
            if not np.isfinite(v) or v <= 0:
                ok = False
                break
            y.append(float(v))

        if not ok:
            out_rows.append({
                "MD": md[i], "INC": float(inc[i]),
                "DTB_Target_Sensor_ft": np.nan, "DTB_Target_Bit_ft": np.nan,
                "R0_ohm_m": np.nan, "Rsh_ohm_m": np.nan,
                "Misfit": np.nan, "Confidence": 0.0, "Mode": mode
            })
            continue

        if state_prev is None:
            r0 = guess_R0(row)
            x0 = [r0, r0, max(ch["h0_ft"] for ch in used)]
        else:
            x0 = state_prev

        x_hat, cost = solve_point(np.asarray(y), used, x0)
        dof = max(len(used) - 3, 1)
        chi2 = 2.0 * cost / dof
        conf = float(np.exp(-0.5 * chi2))
        conf = max(0.0, min(1.0, conf))

        state_prev = x_hat.tolist()
        R0, Rsh, h = x_hat
        h_bit = dtbb_ft(h, dbb_ft, inc[i], dip_app_deg)

        out_rows.append({
            "MD": md[i], "INC": float(inc[i]),
            "DTB_Target_Sensor_ft": float(h),
            "DTB_Target_Bit_ft": float(h_bit),
            "R0_ohm_m": float(R0),
            "Rsh_ohm_m": float(Rsh),
            "Misfit": float(chi2),
            "Confidence": conf,
            "Mode": mode
        })

    return pd.DataFrame(out_rows)
