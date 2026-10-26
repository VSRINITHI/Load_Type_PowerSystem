# =====================================================================================
# load_type_pipeline.py – cleaning, features, baseline & inference for Load_Type model.
# Imported by the training notebook and shipped with the saved model bundle.
# =====================================================================================
import warnings
import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

CLASSES  = ["Light_Load", "Medium_Load", "Maximum_Load"]
LABEL2ID = {c: i for i, c in enumerate(CLASSES)}
RENAME = {"Date_Time": "timestamp", "Usage_kWh": "usage_kwh",
          "Lagging_Current_Reactive.Power_kVarh": "lag_kvarh",
          "Leading_Current_Reactive_Power_kVarh": "lead_kvarh",
          "CO2(tCO2)": "co2", "Lagging_Current_Power_Factor": "lag_pf",
          "Leading_Current_Power_Factor": "lead_pf", "NSM": "nsm_raw", "Load_Type": "load_type"}
SENSORS  = ["usage_kwh", "lag_kvarh", "lead_kvarh", "co2", "lag_pf", "lead_pf"]
CAP_COLS = ["usage_kwh", "lag_kvarh", "lead_kvarh", "co2"]
UNBOUNDED = 1e12
VALID_RANGES = {"usage_kwh": (0.0, UNBOUNDED), "lag_kvarh": (0.0, UNBOUNDED), "lead_kvarh": (0.0, UNBOUNDED),
                "co2": (0.0, UNBOUNDED), "lag_pf": (0.0, 100.0), "lead_pf": (0.0, 100.0)}
WINTER_MONTHS    = [11, 12, 1, 2]       # Nov-Feb tariff season (identified in the EDA)
MIN_HISTORY_ROWS = 96                   # 24 h of 15-min readings (size of the longest rolling window)

CAL  = ["slot", "dow", "is_weekend", "is_winter", "slot_sin", "slot_cos"]
ELEC = SENSORS + ["reactive_ratio", "n_missing_raw"]
TEMP = ["usage_lag1", "usage_lag4", "usage_diff1", "usage_roll4_mean", "usage_roll4_std",
        "usage_roll16_mean", "usage_roll96_mean"]
ALL  = CAL + ELEC + TEMP


def normalise_schema(df):
    # Accept original CSV names or already-normalised names; validate required columns.
    d = df.rename(columns=RENAME).copy()
    missing = [c for c in ["timestamp"] + SENSORS if c not in d.columns]
    if missing:
        inv = {v: k for k, v in RENAME.items()}
        missing = [inv.get(c, c) for c in missing]
        raise ValueError(f"Input is missing required column(s) {missing}. "
                         f"Expected the original CSV columns {list(RENAME)[:7]} (or their normalised names).")
    return d


def clean(raw_df):
    # Deterministic row-wise cleaning (no statistics learned from data -> safe before any split).
    d = normalise_schema(raw_df)
    d["timestamp"] = pd.to_datetime(d["timestamp"], format="%d-%m-%Y %H:%M")
    for c in SENSORS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["_na"] = d[SENSORS].isna().sum(axis=1)
    d = (d.sort_values(["timestamp", "_na"]).drop_duplicates("timestamp", keep="first")
           .drop(columns="_na").sort_values("timestamp").reset_index(drop=True))
    for c, (lo, hi) in VALID_RANGES.items():            # one consistent validity rule per column
        d.loc[(d[c] < lo) | (d[c] > hi), c] = np.nan
    d = d.drop(columns="nsm_raw", errors="ignore")      # re-derived from the timestamp as `slot`
    if "load_type" in d.columns:
        d["y"] = d["load_type"].map(LABEL2ID)
    return d


def make_features(d, caps):
    # All operations are causal (current / past rows only) -> safe to compute on the full series.
    f = pd.DataFrame(index=d.index)
    t = d["timestamp"]
    f["slot"]       = t.dt.hour * 4 + t.dt.minute // 15
    f["dow"]        = t.dt.dayofweek
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["is_winter"]  = t.dt.month.isin(WINTER_MONTHS).astype(int)
    f["slot_sin"]   = np.sin(2 * np.pi * f["slot"] / 96)
    f["slot_cos"]   = np.cos(2 * np.pi * f["slot"] / 96)
    s = d[SENSORS].copy()
    s[CAP_COLS] = s[CAP_COLS].clip(upper=caps, axis=1)  # caps fitted on TRAIN only
    f["n_missing_raw"] = s.isna().sum(axis=1)
    s = s.ffill(limit=4)                                # causal gap fill, max 1 hour
    for c in SENSORS:
        f[c] = s[c]
    f["reactive_ratio"] = s["lag_kvarh"] / (s["usage_kwh"] + 1)
    u = s["usage_kwh"]
    f["usage_lag1"]        = u.shift(1)
    f["usage_lag4"]        = u.shift(4)
    f["usage_diff1"]       = u - u.shift(1)
    f["usage_roll4_mean"]  = u.rolling(4,  min_periods=1).mean()
    f["usage_roll4_std"]   = u.rolling(4,  min_periods=2).std()
    f["usage_roll16_mean"] = u.rolling(16, min_periods=1).mean()
    f["usage_roll96_mean"] = u.rolling(96, min_periods=1).mean()
    return f


def check_history(d, min_rows=MIN_HISTORY_ROWS):
    # Warn (never reject) when history is short or the 15-min grid has gaps; flag rows with a full 24 h window.
    t = d["timestamp"]
    n_gaps = int((t.diff().dropna() != pd.Timedelta("15min")).sum())
    if len(d) < min_rows:
        warnings.warn(f"Only {len(d)} readings supplied (< {min_rows} = 24 h). Rolling 24 h features are computed "
                      f"on a partial window; predictions at the start are less reliable.")
    elif n_gaps:
        warnings.warn(f"{n_gaps} gap(s)/irregular step(s) in the 15-min time grid; rolling features assume a regular grid.")
    return ((t - t.shift(min_rows - 1)) == pd.Timedelta(minutes=15 * (min_rows - 1))).rename("full_24h_history")


class RuleBaseline(ClassifierMixin, BaseEstimator):
    # Majority label per (time-slot, season, weekday) – learned from the training rows only.
    def fit(self, X, y):
        d = pd.DataFrame({"s": np.asarray(X["slot"]), "w": np.asarray(X["is_winter"]),
                          "d": np.asarray(X["dow"]), "y": np.asarray(y)})
        mode = lambda s: s.value_counts().idxmax()
        self.full_ = d.groupby(["s", "w", "d"])["y"].agg(mode).to_dict()
        self.slot_ = d.groupby(["s", "w"])["y"].agg(mode).to_dict()
        self.global_ = int(mode(d["y"]))
        self.classes_ = np.unique(d["y"])
        return self

    def predict(self, X):
        out = []
        for s, w, dd in zip(X["slot"], X["is_winter"], X["dow"]):
            out.append(self.full_.get((s, w, dd), self.slot_.get((s, w), self.global_)))
        return np.array(out)


def predict_from_raw(raw_df, bundle):
    # End-to-end inference: ORIGINAL CSV rows -> predicted Load_Type (+ probabilities, + history flag).
    d = clean(raw_df)
    full = check_history(d)
    X = make_features(d, bundle["caps"])[bundle["features"]]
    proba = bundle["model"].predict_proba(X)
    out = d[["timestamp"]].copy()
    out["predicted_load_type"] = [bundle["classes"][i] for i in proba.argmax(1)]
    for i, c in enumerate(bundle["classes"]):
        out[f"p_{c}"] = proba[:, i].round(4)
    out["full_24h_history"] = full.values
    return out


def load_bundle(path):
    b = joblib.load(path)
    if b["versions"]["scikit-learn"] != sklearn.__version__:
        warnings.warn(f"Model trained with scikit-learn {b['versions']['scikit-learn']}, "
                      f"running {sklearn.__version__}: results may differ.")
    return b
