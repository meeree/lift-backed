
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
    """
    Marker points should include:
    - raw logged entries
    - same-day inferred PR combinations before the global envelope is applied

    They should NOT include extra envelope-filled cells.
    """
    marker_rows = []

    for date_value, day_df in df.groupby("date", sort=True):
        day_df = day_df.copy()
        day_df["weight"] = pd.to_numeric(day_df["weight"], errors="coerce")
        day_df["sets"] = pd.to_numeric(day_df["sets"], errors="coerce")
        day_df["reps"] = pd.to_numeric(day_df["reps"], errors="coerce")
        day_df = day_df.dropna(subset=["weight", "sets", "reps"])

        if day_df.empty:
            continue

        # Raw logged points.
        for _, row in day_df.iterrows():
            marker_rows.append(
                {
                    "date": date_value,
                    "weight": float(row["weight"]),
                    "sets": int(row["sets"]),
                    "reps": int(row["reps"]),
                }
            )

        # Same-day inferred PR combinations (pre-envelope only).
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


def my_grid(df: pd.DataFrame, vmin=None, vmax=None, increment=10, title="PR Bingo Chart"):
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

    envelope = np.nan_to_num(vals.copy(), nan=0.0)
    for i in range(1, vals.shape[0] + 1):
        for j in range(1, vals.shape[1] + 1):
            envelope[:i, :j] = np.maximum(envelope[i - 1, j - 1], envelope[:i, :j])

    finite_vals = envelope[np.isfinite(envelope)]
    positive_vals = finite_vals[finite_vals > 0]
    if positive_vals.size == 0:
        auto_vmin, auto_vmax = 0, 100
    else:
        auto_vmin = int(np.floor(positive_vals.min() / increment) * increment)
        auto_vmax = int(np.ceil(positive_vals.max() / increment) * increment)

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

    ax2.plot(x, max_weight, color=weight_color, linewidth=1.9, alpha=0.9)
    ax2.bar(x, max_weight, width=bar_width, color=weight_fill, edgecolor=weight_color, linewidth=1.2, alpha=0.9)
    ax1.set_ylabel("Max Weight", color=text_color, fontsize=11)

    ax2.bar(x, volume, width=bar_width, color=volume_fill, edgecolor=volume_color, linewidth=1.2, alpha=0.9)
    ax2.plot(x, volume, color=volume_color, linewidth=1.9, alpha=0.9)
    ax2.set_ylabel("Volume", color=text_color, fontsize=11)

    if len(x) == 0:
        x_ticks = np.array([])
    else:
        x_ticks = x
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
    fig.tight_layout()#rect=[0, 0, 1, 0.95])
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


if __name__ == "__main__":
    app.run(debug=True)
