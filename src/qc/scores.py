import numpy as np

def _mad(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    return np.nanmedian(np.abs(x - med))

def spike_score(series, window=21):
    x = np.asarray(series, dtype=float)
    n = len(x)
    out = np.zeros(n, dtype=float)
    half = window // 2
    eps = 1e-9
    for i in range(n):
        a = max(0, i-half)
        b = min(n, i+half+1)
        w = x[a:b]
        med = np.nanmedian(w)
        m = _mad(w)
        out[i] = 0.0 if (not np.isfinite(m) or m < eps) else abs(x[i]-med)/(m+eps)
    return out

def horn_score(df, inc_deg, dip_app_deg, horn_angle_gate_deg=55.0, contrast_min=5.0,
               phase_deep='RPD2', phase_shallow='RPS2', att_deep='RAD2'):
    inc = np.asarray(inc_deg, dtype=float)
    alpha = np.abs(inc - (90.0 - float(dip_app_deg)))

    sp_pd = spike_score(df[phase_deep].values) if phase_deep in df.columns else np.zeros(len(df))
    sp_ps = spike_score(df[phase_shallow].values) if phase_shallow in df.columns else np.zeros(len(df))
    sp_ad = spike_score(df[att_deep].values) if att_deep in df.columns else np.zeros(len(df))

    if 'R0_ohm_m' in df.columns and 'Rsh_ohm_m' in df.columns:
        r0 = np.asarray(df['R0_ohm_m'], dtype=float)
        rsh = np.asarray(df['Rsh_ohm_m'], dtype=float)
        contrast = np.maximum(r0, rsh) / np.maximum(np.minimum(r0, rsh), 1e-6)
    else:
        contrast = np.ones(len(df))

    a_gate = np.clip((alpha - horn_angle_gate_deg) / 20.0, 0.0, 1.0)
    c_gate = np.clip((np.log10(np.maximum(contrast,1.0)) - np.log10(contrast_min)) / (np.log10(50.0)-np.log10(contrast_min)), 0.0, 1.0)
    spike_gate = np.clip((sp_pd - 8.0)/6.0, 0.0, 1.0) * np.clip((sp_pd - sp_ps)/8.0, 0.0, 1.0)
    pa_gate = np.clip((sp_pd - sp_ad)/6.0 + 0.2, 0.0, 1.0) if att_deep in df.columns else np.ones(len(df))

    return a_gate * c_gate * spike_gate * pa_gate

def dielectric_score(df, phase2='RPD2', att2='RAD2', phase4='RPD4', att4='RAD4'):
    n = len(df)
    def slog(a):
        a = np.asarray(a, dtype=float)
        return np.log(np.maximum(a, 1e-6))

    d2 = np.zeros(n)
    d4 = np.zeros(n)
    if phase2 in df.columns and att2 in df.columns:
        d2 = slog(df[att2]) - slog(df[phase2])
    if phase4 in df.columns and att4 in df.columns:
        d4 = slog(df[att4]) - slog(df[phase4])

    raw = d2 - d4
    return np.clip((raw - 0.2)/0.6, 0.0, 1.0)

def anisotropy_score(df, inc_deg, deep='RPD2', shallow='RPS2', window=61):
    n = len(df)
    inc = np.asarray(inc_deg, dtype=float)
    if deep not in df.columns or shallow not in df.columns:
        return np.zeros(n)
    sep = np.log(np.maximum(df[deep].values,1e-6)) - np.log(np.maximum(df[shallow].values,1e-6))

    half = window//2
    corr = np.zeros(n)
    for i in range(n):
        a = max(0, i-half)
        b = min(n, i+half+1)
        x = inc[a:b]
        y = sep[a:b]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            corr[i] = 0.0
            continue
        x = x[m]; y = y[m]
        x = (x-x.mean())/(x.std()+1e-9)
        y = (y-y.mean())/(y.std()+1e-9)
        corr[i] = float(np.mean(x*y))

    return np.clip((corr - 0.3)/0.4, 0.0, 1.0)

def invasion_score(df, mud_system='WBM', shallow='RPS2', deep='RPD2'):
    n = len(df)
    if shallow not in df.columns or deep not in df.columns:
        return np.zeros(n), np.array(['NA']*n)

    rs = df[shallow].astype(float).values
    rd = df[deep].astype(float).values
    sep = np.abs(np.log(np.maximum(rd,1e-6)) - np.log(np.maximum(rs,1e-6)))
    base = np.clip((sep - 0.1)/0.5, 0.0, 1.0)
    score = base if str(mud_system).upper().startswith('OBM') else base*0.7
    pattern = np.where(rd>rs, 'Deep>Shallow', np.where(rs>rd, 'Shallow>Deep', 'Equal'))
    return score, pattern
