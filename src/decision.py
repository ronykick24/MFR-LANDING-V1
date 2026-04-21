import pandas as pd

def decision_events(out: pd.DataFrame, warn_ft: float, action_ft: float, stop_ft: float):
    rows = []
    last = None
    for _, r in out.iterrows():
        h = r.get("DTB_Target_Bit_ft")
        if pd.isna(h):
            continue
        md = float(r["MD"])
        c = float(r.get("Confidence", 0.0))
        mode = str(r.get("Mode", "below"))

        state = None
        if h <= stop_ft:
            state = "STOP"
        elif h <= action_ft:
            state = "ACTION"
        elif h <= warn_ft:
            state = "WARN"

        if state and state != last:
            rows.append({"MD": md, "Mode": mode, "Event": state, "DTB_Target_Bit_ft": float(h), "Confidence": c})
            last = state

    return pd.DataFrame(rows)
