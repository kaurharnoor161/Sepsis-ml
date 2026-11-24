import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def _div(a, b):
    return float(a) / float(b) if b else 0.0


def _share(x):
    return float(np.mean(x)) if len(x) else float("nan")


def _median(x):
    return float(np.median(x)) if len(x) else float("nan")


def row_metrics(y, score, thr):
    y = np.asarray(y).astype(int)
    alert = np.asarray(score) >= thr
    pos = y == 1
    tp = int((alert & pos).sum())
    fp = int((alert & ~pos).sum())
    fn = int((~alert & pos).sum())
    tn = int((~alert & ~pos).sum())
    sens = _div(tp, tp + fn)
    spec = _div(tn, tn + fp)
    prec = _div(tp, tp + fp)
    n = len(y)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": _div(2 * prec * sens, prec + sens),
        "f2": _div(5 * prec * sens, 4 * prec + sens),
        "accuracy": _div(tp + tn, n),
        "alert_rate": _div(tp + fp, n),
    }


def curve_metrics(y, score):
    y = np.asarray(y).astype(int)
    s = np.asarray(score)
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}


def best_f2_threshold(y, score):
    y = np.asarray(y).astype(int)
    s = np.asarray(score)
    cand = np.unique(s)
    if len(cand) > 5000:
        cand = np.unique(np.quantile(s, np.linspace(0, 1, 5000), method="higher"))
    all_sorted = np.sort(s)
    pos_sorted = np.sort(s[y == 1])
    # counts at each candidate threshold without looping over rows
    n_alert = len(all_sorted) - np.searchsorted(all_sorted, cand, side="left")
    tp = len(pos_sorted) - np.searchsorted(pos_sorted, cand, side="left")
    fp = n_alert - tp
    fn = len(pos_sorted) - tp
    den = 5 * tp + 4 * fn + fp
    f2 = np.where(den > 0, 5 * tp / np.maximum(den, 1), 0.0)
    ties = np.flatnonzero(f2 >= f2.max() - 1e-12)
    return float(cand[ties].max())


def sweep_thresholds(y, score, thresholds):
    rows = [{"threshold": float(t), **row_metrics(y, score, t)} for t in thresholds]
    return pd.DataFrame(rows)


def alert_metrics(meta, score, thr):
    hours = np.asarray(meta["hour"]).astype(np.int64)
    ids = np.asarray(meta["patient_id"])
    alert = np.asarray(score) >= thr
    g = meta.groupby("patient_id", sort=False)
    pt = g[["ever_septic", "first_positive_hour"]].first()
    n_rows = g.size().reindex(pt.index).to_numpy()
    septic = pt["ever_septic"].astype(int).to_numpy() == 1
    fph = pt["first_positive_hour"].to_numpy(dtype=float)
    # first alert hour per patient, only patients with alert rows
    a = pd.DataFrame({"patient_id": ids[alert], "hour": hours[alert]})
    first = a.groupby("patient_id")["hour"].min().reindex(pt.index).to_numpy(dtype=float)
    hit = ~np.isnan(first)
    early = septic & (fph > 0)
    first_row = septic & (fph == 0)
    never = ~septic
    by_label = hit & (first <= fph)
    lead = (fph - first)[early & by_label]
    timing = (fph - first)[early & hit]
    # alert episodes in never-septic patients, 6 h apart at least
    eps = 0
    an = a[a["patient_id"].isin(pt.index[never])]
    for _, h in an.groupby("patient_id")["hour"]:
        last = -10**9
        for x in np.sort(h.to_numpy()):
            if x - last >= 6:
                eps += 1
                last = x
    pat_hours = float(n_rows[never].sum())
    return {
        "n_early": int(early.sum()),
        "share_alerted_by_first_label": _share(by_label[early]),
        "share_alerted_any": _share(hit[early]),
        "median_lead_h": _median(lead),
        "median_timing_all_alerted_h": _median(timing),
        "n_first_row": int(first_row.sum()),
        "share_alerted_any_first_row": _share(hit[first_row]),
        "n_nonseptic": int(never.sum()),
        "share_nonseptic_ever_alerted": _share(hit[never]),
        "false_alert_episodes_per_100_patient_days": eps / (pat_hours / 2400) if pat_hours else float("nan"),
    }
