import pandas as pd

from ..data import store


def record_recommendations(conn, rec_date, recs):
    rows = []
    for horizon, items in recs.items():
        for r in items:
            rows.append((rec_date, r.code, r.name, r.horizon, r.strategy,
                         r.reason, r.price, r.position_pct, r.stop_price,
                         r.exit_hint))
    conn.executemany(
        "INSERT INTO recommendations(rec_date, code, name, horizon, strategy,"
        " reason, entry_price, position_pct, stop_price, exit_hint)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()


def evaluate(conn):
    recs = pd.read_sql("SELECT * FROM recommendations", conn)
    rows = []
    for r in recs.itertuples():
        bars = store.load_bars(conn, r.code)
        after = bars[bars["date"] > pd.Timestamp(r.rec_date)]
        if after.empty:
            continue
        latest = float(after["close"].iloc[-1])
        rows.append({"code": r.code, "horizon": r.horizon, "strategy": r.strategy,
                     "rec_date": r.rec_date, "ret": latest / r.entry_price - 1,
                     "days": len(after)})
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"overall": {"n": 0, "hit_rate": 0.0, "avg_ret": 0.0},
                "by_horizon": {}, "detail": detail}

    def summarize(g):
        return {"n": len(g), "hit_rate": float((g["ret"] > 0).mean()),
                "avg_ret": float(g["ret"].mean())}

    return {"overall": summarize(detail),
            "by_horizon": {h: summarize(g) for h, g in detail.groupby("horizon")},
            "detail": detail}
