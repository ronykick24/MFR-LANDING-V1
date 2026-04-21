import numpy as np

def w_logistic(h, h0, s):
    x = (float(h0) - float(h)) / max(float(s), 1e-6)
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(x))

def predict_channel(R0, Rsh, h, h0, s):
    w = w_logistic(h, h0, s)
    return w * float(R0) + (1.0 - w) * float(Rsh)
