from io import BytesIO
from flask_cors import CORS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from flask import Flask, Response, render_template, request
import numpy as np

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


def infer_daily_pr_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each day, infer aggregate achievements from multiple entries.

    Example:
    - 1x3 @ 365
    - 2x3 @ 385
    - 1x4 @ 465

    implies on that date:
    - 4x3 @ 365
    - 3x3 @ 385
    - 1x4 @ 465

    The plotting layer will then keep only the best weights for each sets x reps cell,
    and the existing envelope logic will still fill easier cells.
    """
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


def my_grid(df, vmin=None, vmax=None, increment=10, title="PR Bingo Chart"):
    max_sets = max(10, int(df["sets"].max()))
    max_reps = max(8, int(df["reps"].max()))
    sets = np.arange(1, max_sets + 1)
    reps = np.arange(1, max_reps + 1)
    grid = np.stack(np.meshgrid(reps, sets))
    vals = np.full(grid[0].shape, np.nan, dtype=float)

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
        if s >= 1 and r >= 1 and s <= vals.shape[0] and r <= vals.shape[1]:
            vals[s - 1, r - 1] = float(row["weight"])

    record = []
    times = []

    def add_marker(s, r, v, dt):
        record.append([s - 1, r - 1, v])
        time = dt.timetuple().tm_yday + 365 * (dt.year - 2025)
        times.append(time)

    marker_df = (
        df.groupby(["date", "sets", "reps"], as_index=False)["weight"]
        .max()
        .sort_values("date")
    )
    for _, row in marker_df.iterrows():
        add_marker(int(row["sets"]), int(row["reps"]), float(row["weight"]), row["date"])

    times = np.array(times, dtype=float)
    if len(times) > 0 and times.max() > 0:
        times = times / times.max()

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


@app.route("/plot.png", methods=["POST"])
def plot_png():
    try:
        data = request.get_json()

        if not data or "lifts" not in data:
            return Response("Missing lift data.", status=400)

        lifts = data["lifts"]
        if not lifts:
            return Response("No lifts provided.", status=400)

        controls = data.get("controls", {})
        lift_name = data.get("lift_name", "PR Bingo Chart")

        vmin = controls.get("vmin")
        vmax = controls.get("vmax")
        increment = controls.get("increment", 10)

        vmin = None if vmin in ("", None) else float(vmin)
        vmax = None if vmax in ("", None) else float(vmax)
        increment = 10 if increment in ("", None) else float(increment)

        if increment <= 0:
            return Response("Increment must be positive.", status=400)

        df = pd.DataFrame(lifts)

        required_cols = {"date", "weight", "sets", "reps"}
        if not required_cols.issubset(df.columns):
            return Response("Lift data missing required fields.", status=400)

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["sets"] = pd.to_numeric(df["sets"], errors="coerce")
        df["reps"] = pd.to_numeric(df["reps"], errors="coerce")

        df = df.dropna(subset=["date", "weight", "sets", "reps"]).copy()
        df = df[(df["weight"] > 0) & (df["sets"] > 0) & (df["reps"] > 0)]

        if df.empty:
            return Response("No valid lift data after parsing.", status=400)

        df["sets"] = df["sets"].astype(int)
        df["reps"] = df["reps"].astype(int)
        df = df.sort_values("date")

        fig = my_grid(df, vmin=vmin, vmax=vmax, increment=increment, title=lift_name)

        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150)
        plt.close(fig)
        buf.seek(0)

        return Response(buf.getvalue(), mimetype="image/png")

    except Exception as e:
        return Response(f"Error generating plot: {str(e)}", status=500)


if __name__ == "__main__":
    app.run(debug=True)
