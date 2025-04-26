import pandas as pd
import yfinance as yf

def compute_forward_returns(analog_dates: list[pd.Timestamp],
                            horizons: list[int] = [1,5,20]) -> pd.DataFrame:
    """
    For each analog date, compute the forward returns over calendar‐day horizons,
    mapping to the next trading day via .bfill(). Returns a DataFrame with
    columns: 'date', 'horizon', 'return_pct'.
    """
    # 1) make sure dates are naive and normalized to midnight
    analog_dates = [pd.to_datetime(d).tz_localize(None).normalize() for d in analog_dates]

    # 2) Grab extra history around your analogs
    start = min(analog_dates) - pd.Timedelta(days=30)
    end   = max(analog_dates) + pd.Timedelta(days=max(horizons)+5)
    hist = yf.Ticker('NVDA').history(start=start, end=end)['Close']

    # 3) Drop timezone and normalize index
    hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

    records = []
    for d in analog_dates:
        # find the trading-day at or after d
        idx0 = hist.index.get_indexer([d], method='bfill')[0]
        if idx0 == -1 or idx0 >= len(hist):
            continue
        t0_date = hist.index[idx0]
        price_t0 = hist.iloc[idx0]

        for h in horizons:
            target = d + pd.Timedelta(days=h)
            idx1 = hist.index.get_indexer([target], method='bfill')[0]
            if idx1 == -1 or idx1 >= len(hist):
                continue
            t1_date = hist.index[idx1]
            price_t1 = hist.iloc[idx1]
            ret = (price_t1 / price_t0 - 1) * 100
            records.append({
                'date': t0_date,
                'horizon': h,
                'return_pct': ret
            })

    return pd.DataFrame(records)
