"""Check that MachineLearningCSV rows align 1:1 with TrafficLabelling rows."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generator import CIC_DAYS_ORDER, _safe_read_csv

ML = 'data/CIC-IDS2017/MachineLearningCSV'
TL = 'data/CIC-IDS2017/TrafficLabelling'
for f in CIC_DAYS_ORDER:
    a = _safe_read_csv(os.path.join(ML, f)); a.columns = [c.strip() for c in a.columns]
    b = _safe_read_csv(os.path.join(TL, f)); b.columns = [c.strip() for c in b.columns]
    b = b.dropna(how='all')
    print(f, len(a), len(b))
    n = min(len(a), len(b))
    for col in ('Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Label'):
        eq = (a[col].iloc[:n].astype(str).values == b[col].iloc[:n].astype(str).values).mean()
        print(f'   {col:<20} equal fraction {eq:.6f}')
