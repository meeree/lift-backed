from io import BytesIO

from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request


app = Flask(__name__)
CORS(app)

METRIC_OPTIONS = {
    "log_linear_surface": "Frontier Log Surface",
    "fatigue_exponential": "Fatigue Exponential",
    "separable_log_surface": "Separable Log Surface",
}


@app.route("/")
def index():
    return render_template("index.html")


def infer_daily_pr_rows(df: pd.DataFrame) -> pd.DataFrame:
    inferred_rows = []

    for date_value, day_df in df.groupby("date", sort=True):
        day_df = day_df.copy()
        day_df["weight"] = pd.to_numeric(day_df["weight"], errors="coerce")
        day_df["sets"] = pd.to_numeric(day_df["sets"], errors="coerce")
        day_df["reps"] = pd.to_numeric(day_df["reps"], errors="coerce")
        day_df = day_df.dropna(subset=["weight", "sets", "reps"])

        if day_df.empty:
            continue

        candidate_weights = sorted(day_df["weight"].unique())
        max_reps = int(day_df["reps"].max())

        for weight_threshold in candidate_weights:
            qualifying_for_weight = day_df[day_df["weight"] >= weight_threshold]
            if qualifying_for_weight.empty:
                continue

            for rep_threshold in range(1, max_reps + 1):
                total_sets = int(
                    qualifying_for_weight.loc[
                        qualifying_for_weight["reps"] >= rep_threshold, "sets"
                    ].sum()
                )

                if total_sets <= 0:
                    continue

                inferred_rows.append(
                    {
                        "date": date_value,
                        "weight": float(weight_threshold),
                        "sets": total_sets,
                        "reps": rep_threshold,
                    }
                )

    if not inferred_rows:
        return pd.DataFrame(columns=["date", "weight", "sets", "reps"])

    inferred_df = pd.DataFrame(inferred_rows)
    inferred_df = (
        inferred_df.groupby(["date", "sets", "reps"], as_index=False)["weight"]
        .max()
        .sort_values(["date", "sets", "reps", "weight"])
    )
    return inferred_df


def compute_marker_points(df: pd.DataFrame) -> pd.DataFrame:
    marker_rows = []

    for date_value, day_df in df.groupby("date", sort=True):
        day_df = day_df.copy()
        day_df["weight"] = pd.to_numeric(day_df["weight"], errors="coerce")
        day_df["sets"] = pd.to_numeric(day_df["sets"], errors="coerce")
        day_df["reps"] = pd.to_numeric(day_df["reps"], errors="coerce")
        day_df = day_df.dropna(subset=["weight", "sets", "reps"])

        if day_df.empty:
            continue

        for _, row in day_df.iterrows():
            marker_rows.append(
                {
                    "date": date_value,
                    "weight": float(row["weight"]),
                    "sets": int(row["sets"]),
                    "reps": int(row["reps"]),
                }
            )

        combos = {}
        for _, row in day_df.iterrows():
            weight_threshold = float(row["weight"])
            rep_threshold = int(row["reps"])

            qualifying = day_df[
                (day_df["weight"] >= weight_threshold)
                & (day_df["reps"] >= rep_threshold)
            ]
            total_sets = int(qualifying["sets"].sum())

            if total_sets <= 0:
                continue

            key = (total_sets, rep_threshold)
            if key not in combos or combos[key] < weight_threshold:
                combos[key] = weight_threshold

        for (sets_value, reps_value), weight_value in combos.items():
            marker_rows.append(
                {
                    "date": date_value,
                    "weight": float(weight_value),
                    "sets": int(sets_value),
                    "reps": int(reps_value),
                }
            )

    if not marker_rows:
        return pd.DataFrame(columns=["date", "weight", "sets", "reps"])

    marker_df = pd.DataFrame(marker_rows)
    marker_df = (
        marker_df.groupby(["date", "sets", "reps"], as_index=False)["weight"]
        .max()
        .sort_values(["date", "sets", "reps", "weight"])
    )
    return marker_df


def compute_normalized_session_times(dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    unique_dates = pd.Series(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    if unique_dates.empty:
        return np.array([]), np.array([])

    ordinals = unique_dates.map(pd.Timestamp.toordinal).astype(float).to_numpy()
    if len(ordinals) == 1:
        normalized = np.array([0.0])
    else:
        normalized = (ordinals - ordinals.min()) / (ordinals.max() - ordinals.min())

    return unique_dates.to_numpy(), normalized



def compute_weekly_tick_positions(dates: pd.Series) -> tuple[np.ndarray, list[str]]:
    parsed = pd.to_datetime(pd.Series(dates).dropna()).dt.normalize()
    if parsed.empty:
        return np.array([]), []

    start = parsed.min()
    end = parsed.max()
    if start == end:
        return np.array([0.0]), [start.strftime("%b %-d")]

    start_week = start - pd.Timedelta(days=(start.weekday() + 1) % 7)
    week_dates = pd.date_range(start=start_week, end=end + pd.Timedelta(days=6), freq="7D")
    week_dates = week_dates[(week_dates >= start) & (week_dates <= end)]
    if len(week_dates) == 0:
        week_dates = pd.DatetimeIndex([start, end])

    start_ord = float(start.toordinal())
    span = float(end.toordinal() - start.toordinal())
    x = np.array([(float(d.toordinal()) - start_ord) / span for d in week_dates], dtype=float)
    labels = [d.strftime("%b %-d") for d in week_dates]
    return x, labels


def compute_continuous_bar_width(x: np.ndarray, default_width: float = 0.035) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return default_width
    diffs = np.diff(np.sort(np.unique(x)))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return default_width
    return float(np.clip(0.62 * diffs.min(), 0.010, 0.050))


def max_envelope_volume_for_lift(lift_df: pd.DataFrame) -> float:
    vals, envelope = build_true_pr_and_envelope(lift_df)
    sets_grid, reps_grid = _grid_sr(envelope.shape)
    mask = observed_envelope_mask(vals) & np.isfinite(envelope) & (envelope > 0)
    if not mask.any():
        return float((lift_df["weight"] * lift_df["sets"] * lift_df["reps"]).max())
    volumes = envelope * sets_grid * reps_grid
    return float(np.nanmax(np.where(mask, volumes, np.nan)))


def build_true_pr_and_envelope(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    max_sets = max(10, int(df["sets"].max()))
    max_reps = max(8, int(df["reps"].max()))
    vals = np.full((max_sets, max_reps), np.nan, dtype=float)

    inferred_df = infer_daily_pr_rows(df)
    if inferred_df.empty:
        inferred_df = df[["date", "weight", "sets", "reps"]].copy()

    cell_best = (
        inferred_df.groupby(["sets", "reps"], as_index=False)["weight"]
        .max()
        .sort_values(["sets", "reps"])
    )

    for _, row in cell_best.iterrows():
        s = int(row["sets"])
        r = int(row["reps"])
        if 1 <= s <= vals.shape[0] and 1 <= r <= vals.shape[1]:
            vals[s - 1, r - 1] = float(row["weight"])

    envelope = np.nan_to_num(vals.copy(), nan=0.0)
    for i in range(1, vals.shape[0] + 1):
        for j in range(1, vals.shape[1] + 1):
            envelope[:i, :j] = np.maximum(envelope[i - 1, j - 1], envelope[:i, :j])

    return vals, envelope


def observed_envelope_mask(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    rows, cols = vals.shape
    mask = np.zeros_like(vals, dtype=bool)
    for i in range(rows):
        for j in range(cols):
            sub = vals[i:, j:]
            mask[i, j] = np.isfinite(sub).any()
    return mask


def frontier_mask_from_matrix(matrix: np.ndarray) -> np.ndarray:
    observed = np.isfinite(matrix) & (matrix > 0)
    frontier = observed.copy()
    rows, cols = matrix.shape

    for s in range(rows):
        for r in range(cols):
            if not observed[s, r]:
                continue
            w = matrix[s, r]
            dominated = False
            for s2 in range(s, rows):
                for r2 in range(r, cols):
                    if s2 == s and r2 == r:
                        continue
                    if observed[s2, r2] and matrix[s2, r2] >= w and (s2 > s or r2 > r):
                        dominated = True
                        break
                if dominated:
                    break
            frontier[s, r] = not dominated

    return frontier


def _grid_sr(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    s = np.arange(1, rows + 1, dtype=float)[:, None]
    r = np.arange(1, cols + 1, dtype=float)[None, :]
    return np.broadcast_to(s, shape), np.broadcast_to(r, shape)


def fit_log_linear_surface_from_envelope(envelope: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    s_idx, r_idx = np.where(fit_mask)
    s = s_idx.astype(float) + 1.0
    r = r_idx.astype(float) + 1.0
    y = envelope[fit_mask]

    X = np.column_stack(
        [
            np.ones_like(s),
            np.log(s),
            np.log(r),
            np.log(s) * np.log(r),
        ]
    )
    y_log = np.log(np.clip(y, 1e-6, None))
    ridge = 1e-3 * np.eye(X.shape[1])
    beta = np.linalg.solve(X.T @ X + ridge, X.T @ y_log)

    sg, rg = _grid_sr(envelope.shape)
    Xg = np.stack(
        [
            np.ones_like(sg),
            np.log(sg),
            np.log(rg),
            np.log(sg) * np.log(rg),
        ],
        axis=-1,
    )
    pred = np.exp(Xg @ beta)
    return np.minimum(pred, envelope[0, 0])


def fit_fatigue_exponential_from_envelope(envelope: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    s_idx, r_idx = np.where(fit_mask)
    s = s_idx.astype(float) + 1.0
    r = r_idx.astype(float) + 1.0
    y = envelope[fit_mask]

    X = np.column_stack(
        [
            np.ones_like(s),
            -(r - 1.0),
            -(s - 1.0),
            -((s - 1.0) * (r - 1.0)),
        ]
    )
    y_log = np.log(np.clip(y, 1e-6, None))
    ridge = 1e-3 * np.eye(X.shape[1])
    beta = np.linalg.solve(X.T @ X + ridge, X.T @ y_log)

    sg, rg = _grid_sr(envelope.shape)
    Xg = np.stack(
        [
            np.ones_like(sg),
            -(rg - 1.0),
            -(sg - 1.0),
            -((sg - 1.0) * (rg - 1.0)),
        ],
        axis=-1,
    )
    pred = np.exp(Xg @ beta)
    return np.minimum(pred, envelope[0, 0])


def fit_separable_log_surface_from_envelope(envelope: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    y_log = np.log(np.clip(envelope[fit_mask], 1e-6, None))
    row_ids, col_ids = np.where(fit_mask)

    n_rows, n_cols = envelope.shape
    row_means = np.zeros(n_rows, dtype=float)
    col_means = np.zeros(n_cols, dtype=float)

    global_mean = y_log.mean()
    for i in range(n_rows):
        vals = y_log[row_ids == i]
        row_means[i] = vals.mean() if len(vals) > 0 else global_mean
    for j in range(n_cols):
        vals = y_log[col_ids == j]
        col_means[j] = vals.mean() if len(vals) > 0 else global_mean

    pred_log = global_mean + (row_means[:, None] - global_mean) + (col_means[None, :] - global_mean)
    pred = np.exp(pred_log)
    return np.minimum(pred, envelope[0, 0])


def compute_metric_prediction(envelope: np.ndarray, fit_mask: np.ndarray, metric_name: str) -> np.ndarray:
    if metric_name == "log_linear_surface":
        pred = fit_log_linear_surface_from_envelope(envelope, fit_mask)
    elif metric_name == "fatigue_exponential":
        pred = fit_fatigue_exponential_from_envelope(envelope, fit_mask)
    elif metric_name == "separable_log_surface":
        pred = fit_separable_log_surface_from_envelope(envelope, fit_mask)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

    pred = np.minimum.accumulate(pred, axis=0)
    pred = np.minimum.accumulate(pred, axis=1)
    return np.clip(pred, 0.0, None)


def auto_color_range(matrix: np.ndarray, increment: float) -> tuple[float, float]:
    finite_vals = matrix[np.isfinite(matrix)]
    positive_vals = finite_vals[finite_vals > 0]
    if positive_vals.size == 0:
        auto_vmin, auto_vmax = 0.0, 100.0
    else:
        auto_vmin = float(np.floor(positive_vals.min() / increment) * increment)
        auto_vmax = float(np.ceil(positive_vals.max() / increment) * increment)
    return auto_vmin, auto_vmax


def compute_metric_diffs(df: pd.DataFrame) -> dict:
    vals, envelope = build_true_pr_and_envelope(df)
    fit_mask = frontier_mask_from_matrix(envelope)
    valid_mask = observed_envelope_mask(vals)

    diffs = {}
    max_abs = 0.0

    for metric_name in METRIC_OPTIONS:
        pred = compute_metric_prediction(envelope, fit_mask, metric_name)
        diff = pred - envelope
        diff[~valid_mask] = np.nan
        diffs[metric_name] = diff
        if np.isfinite(diff).any():
            max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))

    if max_abs <= 0:
        max_abs = 10.0

    return {
        "vals": vals,
        "envelope": envelope,
        "fit_mask": fit_mask,
        "valid_mask": valid_mask,
        "diffs": diffs,
        "max_abs": max_abs,
    }



def _neighbors_8(i: int, j: int, rows: int, cols: int):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                yield ni, nj


def _hill_climb_local_max(score: np.ndarray, start_i: int, start_j: int, valid_mask: np.ndarray):
    i, j = start_i, start_j
    rows, cols = score.shape
    while True:
        best_i, best_j = i, j
        best_val = score[i, j]
        for ni, nj in _neighbors_8(i, j, rows, cols):
            if not valid_mask[ni, nj] or not np.isfinite(score[ni, nj]):
                continue
            if score[ni, nj] > best_val + 1e-9:
                best_i, best_j = ni, nj
                best_val = score[ni, nj]
        if best_i == i and best_j == j:
            return i, j
        i, j = best_i, best_j


def _plateau_impact(envelope: np.ndarray, i: int, j: int) -> int:
    w = envelope[i, j]
    impact = 0
    for ii in range(i + 1):
        for jj in range(j + 1):
            if np.isfinite(envelope[ii, jj]) and abs(envelope[ii, jj] - w) < 1e-9:
                impact += 1
    return impact


def _dedupe_ranked_candidates(candidates: list[dict], top_k: int, min_manhattan: int = 2) -> list[dict]:
    chosen = []
    for cand in candidates:
        keep = True
        for prev in chosen:
            dist = abs(cand["sets"] - prev["sets"]) + abs(cand["reps"] - prev["reps"])
            if dist < min_manhattan:
                keep = False
                break
        if keep:
            chosen.append(cand)
        if len(chosen) >= top_k:
            break
    return chosen


def generate_session_recommendations(df: pd.DataFrame, top_k: int = 10) -> dict:
    data = compute_metric_diffs(df)
    vals = data["vals"]
    envelope = data["envelope"]
    valid_mask = data["valid_mask"]
    metric_names = list(METRIC_OPTIONS.keys())

    positive_diffs = []
    for metric_name in metric_names:
        diff = data["diffs"][metric_name]
        positive_diffs.append(np.where(np.isfinite(diff), np.maximum(diff, 0.0), np.nan))

    stacked = np.stack(positive_diffs, axis=0)
    mean_score = np.nanmean(stacked, axis=0)
    mean_score[~valid_mask] = np.nan

    rng = np.random.default_rng(0)
    valid_coords = np.argwhere(valid_mask & np.isfinite(mean_score))
    low_hanging_candidates = []

    if len(valid_coords) > 0:
        num_seeds = min(300, max(60, 8 * len(valid_coords)))
        seed_indices = rng.integers(0, len(valid_coords), size=num_seeds)
        seen = set()

        for seed_idx in seed_indices:
            si, sj = valid_coords[seed_idx]
            li, lj = _hill_climb_local_max(mean_score, int(si), int(sj), valid_mask)
            key = (li, lj)
            if key in seen:
                continue
            seen.add(key)

            current_weight = float(vals[li, lj]) if np.isfinite(vals[li, lj]) else None
            low_hanging_candidates.append(
                {
                    "sets": int(li + 1),
                    "reps": int(lj + 1),
                    "score": float(mean_score[li, lj]),
                    "current_weight": current_weight,
                }
            )

    low_hanging_candidates.sort(key=lambda x: (-x["score"], x["sets"], x["reps"]))
    low_hanging = _dedupe_ranked_candidates(low_hanging_candidates, top_k=top_k, min_manhattan=2)

    impactful_candidates = []
    rows, cols = mean_score.shape
    for i in range(rows):
        for j in range(cols):
            if not valid_mask[i, j] or not np.isfinite(mean_score[i, j]):
                continue
            plateau_size = _plateau_impact(envelope, i, j)
            impact_score = plateau_size * max(float(mean_score[i, j]), 0.0)
            if impact_score <= 0:
                continue

            current_weight = float(vals[i, j]) if np.isfinite(vals[i, j]) else None
            impactful_candidates.append(
                {
                    "sets": int(i + 1),
                    "reps": int(j + 1),
                    "score": float(impact_score),
                    "mean_score": float(mean_score[i, j]),
                    "plateau_size": int(plateau_size),
                    "current_weight": current_weight,
                }
            )

    impactful_candidates.sort(
        key=lambda x: (-x["score"], -x["plateau_size"], -x["mean_score"], x["sets"], x["reps"])
    )
    impactful = _dedupe_ranked_candidates(impactful_candidates, top_k=top_k, min_manhattan=2)

    return {
        "low_hanging": low_hanging,
        "most_impactful": impactful,
        "mean_score_grid": mean_score.tolist(),
    }


def my_grid(df: pd.DataFrame, vmin=None, vmax=None, increment=10, title="PR Bingo Chart"):
    vals, envelope = build_true_pr_and_envelope(df)
    marker_df = compute_marker_points(df)

    unique_dates, normalized_times = compute_normalized_session_times(marker_df["date"])
    time_lookup = {pd.Timestamp(d).normalize(): t for d, t in zip(unique_dates, normalized_times)}

    record = []
    times = []
    for _, row in marker_df.iterrows():
        s = int(row["sets"])
        r = int(row["reps"])
        if 1 <= s <= vals.shape[0] and 1 <= r <= vals.shape[1]:
            record.append([s - 1, r - 1, float(row["weight"])])
            times.append(time_lookup.get(pd.Timestamp(row["date"]).normalize(), 0.0))

    times = np.array(times, dtype=float)

    auto_vmin, auto_vmax = auto_color_range(envelope, increment)

    if vmin is None:
        vmin = auto_vmin
    if vmax is None:
        vmax = auto_vmax
    if vmax <= vmin:
        vmax = vmin + increment

    fig = plt.figure(figsize=(6, 5))
    cmap = plt.get_cmap("tab20b").copy()
    cmap.set_bad(color="black")

    clipped = envelope.copy()
    clipped[clipped > vmax] = np.nan
    clipped[clipped < vmin] = np.nan

    plt.imshow(
        clipped,
        interpolation="none",
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    plt.yticks(np.arange(0, vals.shape[0]), np.arange(1, vals.shape[0] + 1))
    plt.xticks(np.arange(0, vals.shape[1]), np.arange(1, vals.shape[1] + 1))
    plt.colorbar(ticks=np.arange(vmin + increment / 2, vmax + 3 * increment / 2, increment))

    if len(record) > 0:
        record = np.array(record).T
        plt.scatter(record[1], record[0], c=times, cmap="Wistia", marker="^", s=80)

    plt.xlabel("Reps", fontsize=14)
    plt.ylabel("Sets", fontsize=14)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    return fig




def make_metric_diff_plot(df: pd.DataFrame, title: str = "Metric Over/Under"):
    data = compute_metric_diffs(df)
    metric_names = list(METRIC_OPTIONS.keys())
    diffs = [data["diffs"][metric_name] for metric_name in metric_names]
    max_abs = data["max_abs"]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9), constrained_layout=True)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f2f2f2")

    for ax, metric_name, diff in zip(axes, metric_names, diffs):
        im = ax.imshow(
            diff,
            interpolation="none",
            origin="lower",
            aspect="auto",
            cmap=cmap,
            vmin=-max_abs,
            vmax=max_abs,
        )
        ax.set_xticks(np.arange(0, diff.shape[1]))
        ax.set_xticklabels(np.arange(1, diff.shape[1] + 1))
        ax.set_yticks(np.arange(0, diff.shape[0]))
        ax.set_yticklabels(np.arange(1, diff.shape[0] + 1))
        ax.set_xlabel("Reps", fontsize=11)
        ax.set_ylabel("Sets", fontsize=11)
        ax.set_title(METRIC_OPTIONS[metric_name], fontsize=12)

        for i in range(diff.shape[0]):
            for j in range(diff.shape[1]):
                value = diff[i, j]
                if not np.isfinite(value):
                    continue
                text = f"{int(np.rint(value))}"
                text_color = "black" if abs(value) < 0.55 * max_abs else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=8, color=text_color)

    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("Predicted - Envelope", rotation=90)
    fig.suptitle(title, fontsize=14)
    return fig

def filter_df_by_months_back(df: pd.DataFrame, months_back) -> pd.DataFrame:
    if months_back in (None, "", 0):
        return df

    try:
        months_back = float(months_back)
    except (TypeError, ValueError):
        return df

    if months_back <= 0:
        return df

    max_date = pd.Timestamp(df["date"].max()).normalize()
    cutoff = max_date - pd.DateOffset(months=months_back)
    filtered = df[df["date"] >= cutoff].copy()

    if filtered.empty:
        return df
    return filtered


def make_session_summary_plot(df: pd.DataFrame, months_back=None, title="Session Summary"):
    df_plot = filter_df_by_months_back(df, months_back)

    volume_by_day = (df_plot["weight"] * df_plot["sets"] * df_plot["reps"]).groupby(df_plot["date"]).sum()
    daily = (
        df_plot.groupby("date", as_index=False)
        .agg(max_weight=("weight", "max"))
        .sort_values("date")
    )
    daily["volume"] = daily["date"].map(volume_by_day).astype(float)

    unique_dates, normalized_times = compute_normalized_session_times(daily["date"])
    time_lookup = {pd.Timestamp(d).normalize(): t for d, t in zip(unique_dates, normalized_times)}
    x = np.array([time_lookup[pd.Timestamp(d).normalize()] for d in daily["date"]], dtype=float)

    max_weight = daily["max_weight"].to_numpy(dtype=float)
    volume = daily["volume"].to_numpy(dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(8.2, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12},
    )

    fig.patch.set_facecolor("#171a21")
    panel_color = "#1d2230"
    grid_color = "#3a4458"
    spine_color = "#2a3140"
    text_color = "#e8ecf1"
    muted_text = "#a8b0bd"
    weight_color = "#8fd3ff"
    weight_fill = "#2f6ea5"
    volume_color = "#79d79b"
    volume_fill = "#2f7a53"

    bar_width = compute_continuous_bar_width(x, default_width=0.05)

    for ax in (ax1, ax2):
        ax.set_facecolor(panel_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.tick_params(colors=muted_text, labelsize=9)
        ax.grid(True, axis="y", color=grid_color, alpha=0.35, linewidth=0.8)
        ax.grid(False, axis="x")

    if len(max_weight) > 0:
        ax1.plot(x, max_weight, color=weight_color, linewidth=1.9, alpha=0.9)
        ax1.bar(x, max_weight, width=bar_width, color=weight_fill, edgecolor=weight_color, linewidth=1.2, alpha=0.9)
        ax1.set_ylim(0.8 * max_weight.min(), 1.1 * max_weight.max())

    if len(volume) > 0:
        ax2.bar(x, volume, width=bar_width, color=volume_fill, edgecolor=volume_color, linewidth=1.2, alpha=0.9)
        ax2.plot(x, volume, color=volume_color, linewidth=1.9, alpha=0.9)
        ax2.set_ylim(0.8 * volume.min(), 1.1 * volume.max())

    ax1.set_ylabel("Max Weight", color=text_color, fontsize=11)
    ax2.set_ylabel("Volume", color=text_color, fontsize=11)

    week_x, week_labels = compute_weekly_tick_positions(daily["date"])
    ax2.set_xticks(week_x)
    ax2.set_xticklabels(week_labels, rotation=0, ha="center", color=muted_text)
    ax2.tick_params(axis="x", length=5, color=muted_text)
    ax2.set_xlabel("Week", color=muted_text, fontsize=10)

    if len(x) <= 1:
        x_left, x_right = -0.06, 1.06
    else:
        x_left, x_right = -0.03, 1.03
    ax1.set_xlim(x_left, x_right)
    ax2.set_xlim(x_left, x_right)

    fig.suptitle(title, fontsize=14, color=text_color, y=0.98)
    fig.tight_layout()
    return fig



def parse_all_lifts_df(payload: dict) -> pd.DataFrame:
    lifts = payload.get("all_lifts") or payload.get("lifts") or []
    if not lifts:
        raise ValueError("No lift data provided.")

    df = pd.DataFrame(lifts)
    required_cols = {"lift", "date", "weight", "sets", "reps"}
    if not required_cols.issubset(df.columns):
        raise ValueError("All-lift data missing lift, date, weight, sets, or reps fields.")

    df["lift"] = df["lift"].fillna("").astype(str).str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["sets"] = pd.to_numeric(df["sets"], errors="coerce")
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce")

    df = df.dropna(subset=["date", "weight", "sets", "reps"]).copy()
    df = df[(df["lift"] != "") & (df["weight"] > 0) & (df["sets"] > 0) & (df["reps"] > 0)]
    if df.empty:
        raise ValueError("No valid all-lift data after parsing.")

    df["sets"] = df["sets"].astype(int)
    df["reps"] = df["reps"].astype(int)
    df["volume"] = df["weight"] * df["sets"] * df["reps"]
    return df.sort_values(["date", "lift"])


def make_all_lift_volume_plot(df: pd.DataFrame, months_back=None, title="All-Lift Volume"):
    df_plot = filter_df_by_months_back(df, months_back)

    total_volume_by_lift = df.groupby("lift")["volume"].sum().to_dict()

    daily = (
        df_plot.groupby(["date", "lift"], as_index=False)["volume"]
        .sum()
        .sort_values(["date", "lift"])
    )
    daily = daily[daily["volume"] > 0].copy()
    daily["total_volume"] = daily["lift"].map(total_volume_by_lift)
    daily = daily[(daily["total_volume"].notna()) & (daily["total_volume"] > 0)]
    daily["relative_volume"] = daily["volume"] / daily["total_volume"]

    if daily.empty:
        raise ValueError("No nonzero volume to plot.")

    pivot = daily.pivot(index="date", columns="lift", values="relative_volume").fillna(0.0)
    pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).index]

    dates = pd.Series(pivot.index)
    unique_dates, normalized_times = compute_normalized_session_times(dates)
    time_lookup = {pd.Timestamp(d).normalize(): t for d, t in zip(unique_dates, normalized_times)}
    x = np.array([time_lookup[pd.Timestamp(d).normalize()] for d in pivot.index], dtype=float)

    lift_names = list(pivot.columns)
    n_lifts = len(lift_names)

    legend_rows = int(np.ceil(max(n_lifts, 1) / 2.0))
    extra_height = min(4.4, 0.38 * legend_rows + 0.8)
    fig_height = 5.2 + extra_height
    fig, ax = plt.subplots(figsize=(10.8, fig_height))

    fig.patch.set_facecolor("#171a21")
    ax.set_facecolor("#1d2230")
    grid_color = "#3a4458"
    spine_color = "#2a3140"
    text_color = "#e8ecf1"
    muted_text = "#a8b0bd"

    for spine in ax.spines.values():
        spine.set_color(spine_color)
    ax.tick_params(colors=muted_text, labelsize=9)
    ax.grid(True, axis="y", color=grid_color, alpha=0.35, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    cmap_name = "tab20" if n_lifts <= 20 else "turbo"
    cmap = plt.get_cmap(cmap_name)
    denom = max(n_lifts - 1, 1)
    colors = [cmap(i / denom) for i in range(n_lifts)]

    bottom = np.zeros(len(pivot), dtype=float)
    bar_width = compute_continuous_bar_width(x, default_width=0.05)

    for lift_name, color in zip(lift_names, colors):
        vals = pivot[lift_name].to_numpy(dtype=float)
        ax.bar(
            x,
            vals,
            bottom=bottom,
            width=bar_width,
            color=color,
            edgecolor="#0f1115",
            linewidth=0.35,
            label=lift_name,
        )
        bottom += vals

    ax.set_ylabel("Daily / Total Lift Volume", color=text_color, fontsize=11)
    ax.set_xlabel("Week", color=muted_text, fontsize=10)
    ax.set_title(title, color=text_color, fontsize=14, pad=12)

    week_x, week_labels = compute_weekly_tick_positions(pd.Series(pivot.index))
    ax.set_xticks(week_x)
    ax.set_xticklabels(week_labels, rotation=0, ha="center", color=muted_text)
    ax.tick_params(axis="x", length=5, color=muted_text)

    if len(x) <= 1:
        ax.set_xlim(-0.06, 1.06)
    else:
        ax.set_xlim(-0.03, 1.03)

    ymax = float(bottom.max()) if len(bottom) else 1.0
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1.0)

    legend = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
        frameon=False,
        fontsize=10,
        labelcolor=text_color,
        columnspacing=1.8,
        handlelength=1.7,
        handletextpad=0.65,
        borderaxespad=0.0,
    )
    for text in legend.get_texts():
        text.set_color(text_color)

    bottom_margin = min(0.50, 0.19 + 0.04 * legend_rows)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=bottom_margin)
    return fig


def parse_request_df():
    data = request.get_json()

    if not data or "lifts" not in data:
        return None, None, None, None, Response("Missing lift data.", status=400)

    lifts = data["lifts"]
    if not lifts:
        return None, None, None, None, Response("No lifts provided.", status=400)

    controls = data.get("controls", {})
    lift_name = data.get("lift_name", "PR Bingo Chart")

    vmin = controls.get("vmin")
    vmax = controls.get("vmax")
    increment = controls.get("increment", 10)
    summary_months_back = controls.get("summary_months_back")

    vmin = None if vmin in ("", None) else float(vmin)
    vmax = None if vmax in ("", None) else float(vmax)
    increment = 10 if increment in ("", None) else float(increment)
    summary_months_back = None if summary_months_back in ("", None) else float(summary_months_back)

    if increment <= 0:
        return None, None, None, None, Response("Increment must be positive.", status=400)

    df = pd.DataFrame(lifts)
    required_cols = {"date", "weight", "sets", "reps"}
    if not required_cols.issubset(df.columns):
        return None, None, None, None, Response("Lift data missing required fields.", status=400)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
    df["sets"] = pd.to_numeric(df["sets"], errors="coerce")
    df["reps"] = pd.to_numeric(df["reps"], errors="coerce")

    df = df.dropna(subset=["date", "weight", "sets", "reps"]).copy()
    df = df[(df["weight"] > 0) & (df["sets"] > 0) & (df["reps"] > 0)]

    if df.empty:
        return None, None, None, None, Response("No valid lift data after parsing.", status=400)

    df["sets"] = df["sets"].astype(int)
    df["reps"] = df["reps"].astype(int)
    df = df.sort_values("date")

    controls_out = {
        "vmin": vmin,
        "vmax": vmax,
        "increment": increment,
        "summary_months_back": summary_months_back,
    }
    return df, lift_name, controls_out, data, None


@app.route("/plot.png", methods=["POST"])
def plot_png():
    try:
        df, lift_name, controls, _, error_response = parse_request_df()
        if error_response is not None:
            return error_response

        fig = my_grid(
            df,
            vmin=controls["vmin"],
            vmax=controls["vmax"],
            increment=controls["increment"],
            title=lift_name,
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        return Response(f"Error generating plot: {str(e)}", status=500)


@app.route("/summary.png", methods=["POST"])
def summary_png():
    try:
        df, lift_name, controls, _, error_response = parse_request_df()
        if error_response is not None:
            return error_response

        fig = make_session_summary_plot(
            df,
            months_back=controls["summary_months_back"],
            title=f"{lift_name} Session Summary",
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        return Response(f"Error generating summary plot: {str(e)}", status=500)


@app.route("/metric.png", methods=["POST"])
def metric_png():
    try:
        df, lift_name, controls, _, error_response = parse_request_df()
        if error_response is not None:
            return error_response

        fig = make_metric_diff_plot(df, title=f"{lift_name} Over/Under")

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        return Response(f"Error generating metric plot: {str(e)}", status=500)




@app.route("/all_volume.png", methods=["POST"])
def all_volume_png():
    try:
        data = request.get_json() or {}
        controls = data.get("controls", {})
        summary_months_back = controls.get("summary_months_back")
        summary_months_back = None if summary_months_back in ("", None) else float(summary_months_back)

        df = parse_all_lifts_df(data)
        fig = make_all_lift_volume_plot(
            df,
            months_back=summary_months_back,
            title="All-Lift Daily Volume",
        )

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return Response(buf.getvalue(), mimetype="image/png")
    except Exception as e:
        return Response(f"Error generating all-lift volume plot: {str(e)}", status=500)

@app.route("/recommendations", methods=["POST"])
def recommendations():
    try:
        df, lift_name, controls, _, error_response = parse_request_df()
        if error_response is not None:
            return error_response

        rec_data = generate_session_recommendations(df, top_k=10)
        low_hanging = rec_data["low_hanging"]
        impactful = rec_data["most_impactful"]

        if not low_hanging and not impactful:
            return {"text": "No recommendations available yet.", "lift_name": lift_name}

        lines = ["Low hanging fruit:"]
        for rec in low_hanging:
            current_weight = rec.get("current_weight")
            current_text = f", current {int(round(current_weight))}" if current_weight is not None else ""
            lines.append(
                f'{rec["sets"]}x{rec["reps"]} (score {int(round(rec["score"]))}{current_text})'
            )

        lines.append("")
        lines.append("Most impactful:")
        for rec in impactful:
            current_weight = rec.get("current_weight")
            current_text = f", current {int(round(current_weight))}" if current_weight is not None else ""
            lines.append(
                f'{rec["sets"]}x{rec["reps"]} (score {int(round(rec["score"]))}, area {rec["plateau_size"]}{current_text})'
            )

        return {
            "text": "\n".join(lines),
            "low_hanging": low_hanging,
            "most_impactful": impactful,
            "lift_name": lift_name,
        }
    except Exception as e:
        return Response(f"Error generating recommendations: {str(e)}", status=500)


if __name__ == "__main__":
    app.run(debug=True)
