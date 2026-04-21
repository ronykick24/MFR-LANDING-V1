import numpy as np

def alpha_rel_deg(inc_deg: float, dip_app_deg: float) -> float:
    return float(abs(float(inc_deg) - (90.0 - float(dip_app_deg))))

def d_sensor_to_bit_ft(dbb_ft: float, inc_deg: float, dip_app_deg: float) -> float:
    a = np.deg2rad(alpha_rel_deg(inc_deg, dip_app_deg))
    return float(dbb_ft) * float(np.sin(a))

def dtbb_ft(dtb_sensor_ft: float, dbb_ft: float, inc_deg: float, dip_app_deg: float) -> float:
    return max(float(dtb_sensor_ft) - d_sensor_to_bit_ft(dbb_ft, inc_deg, dip_app_deg), 0.0)

def lead_md_ft(dtb_sensor_ft: float, dbb_ft: float, inc_deg: float, dip_app_deg: float, bit_margin_ft: float = 3.0) -> float:
    a = np.deg2rad(alpha_rel_deg(inc_deg, dip_app_deg))
    s = float(np.sin(a))
    if s <= 1e-9:
        return float("inf")
    d = float(dbb_ft) * s
    need = d + float(bit_margin_ft)
    return max((float(dtb_sensor_ft) - need) / s, 0.0)
