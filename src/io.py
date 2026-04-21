import pandas as pd

def load_csv(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    if 'DEPTH' in df.columns and 'MD' not in df.columns:
        df = df.rename(columns={'DEPTH':'MD'})
    return df
