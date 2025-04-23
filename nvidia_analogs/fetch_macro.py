import pandas as pd
import pandas_datareader.data as web

def fetch_macro(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Retrieve multiple FRED series, then reindex
    over every calendar day from start→end and fill.
    """
    series = {
        'CPI':      'CPIAUCSL',
        'UNRATE':   'UNRATE',
        'FEDFUNDS': 'FEDFUNDS',
        'PCEPI':    'PCEPI',
        'M2SL':     'M2SL',
        'GS10':     'GS10',
        'VIXCLS':   'VIXCLS'
    }

    dfs = []
    for name, code in series.items():
        s = web.DataReader(code, 'fred', start_date, end_date)
        s = s.rename(columns={code: name})
        dfs.append(s)

    raw = pd.concat(dfs, axis=1)
    # 1) build full calendar index
    full_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    # 2) reindex & fill forward/backward
    macro = raw.reindex(full_idx).ffill().bfill()
    macro.index.name = 'date'
    macro = macro.reset_index()
    return macro
