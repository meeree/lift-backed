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


def generate_session_recommendations(df: pd.DataFrame, top_k: int = 8) -> list[dict]:
    data = compute_metric_diffs(df)
    vals = data["vals"]
    valid_mask = data["valid_mask"]
    metric_names = list(METRIC_OPTIONS.keys())

    aggregate = np.zeros_like(vals, dtype=float)
    agreement = np.zeros_like(vals, dtype=float)

    for metric_name in metric_names:
        diff = data["diffs"][metric_name]
        positive = np.where(np.isfinite(diff), np.maximum(diff, 0.0), 0.0)
        aggregate += positive
        agreement += (positive > 0).astype(float)

    # reward agreement across models, but keep it modest
    score = aggregate + 5.0 * np.maximum(agreement - 1.0, 0.0)
    score[~valid_mask] = np.nan

    rows = []
    for i in range(score.shape[0]):
        for j in range(score.shape[1]):
            if not np.isfinite(score[i, j]):
                continue
            rows.append(
                {
                    "sets": int(i + 1),
                    "reps": int(j + 1),
                    "score": float(score[i, j]),
                    "current_weight": float(vals[i, j]) if np.isfinite(vals[i, j]) else None,
                }
            )

    rows.sort(key=lambda x: (-x["score"], x["sets"], x["reps"]))
    return rows[:top_k]


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

    bar_width = 0.018 if len(x) > 1 else 0.05

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

    x_ticks = x if len(x) > 0 else np.array([])
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels([""] * len(x_ticks))
    ax2.set_xlabel("Training Session", color=muted_text, fontsize=10)

    if len(x) <= 1:
        x_left, x_right = -0.06, 1.06
    else:
        x_left, x_right = x.min() - 0.03, x.max() + 0.03
    ax1.set_xlim(x_left, x_right)
    ax2.set_xlim(x_left, x_right)

    fig.suptitle(title, fontsize=14, color=text_color, y=0.98)
    fig.tight_layout()
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


@app.route("/recommendations", methods=["POST"])
def recommendations():
    try:
        df, lift_name, controls, _, error_response = parse_request_df()
        if error_response is not None:
            return error_response

        recs = generate_session_recommendations(df, top_k=8)
        if not recs:
            return {"text": "No recommendations available yet.", "recommendations": []}

        lines = ["Recommended next sessions ranked:"]
        for rec in recs:
            current_weight = rec.get("current_weight")
            current_text = (
                f", current {int(round(current_weight))}"
                if current_weight is not None
                else ""
            )
            lines.append(
                f'{rec["sets"]}x{rec["reps"]} (score {int(round(rec["score"]))}{current_text})'
            )

        return {"text": "\n".join(lines), "recommendations": recs, "lift_name": lift_name}
    except Exception as e:
        return Response(f"Error generating recommendations: {str(e)}", status=500)


if __name__ == "__main__":
    app.run(debug=True)
