import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import os
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np
    import optuna
    import pandas as pd
    from dotenv import load_dotenv
    from scipy.optimize import curve_fit

    load_dotenv(Path(__file__).parents[3] / ".env")

    study_name = "tune_v5"
    return (
        Path,
        alt,
        curve_fit,
        load_dotenv,
        mo,
        np,
        optuna,
        os,
        pd,
        study_name,
    )


@app.cell
def _(mo):
    refresh = mo.ui.refresh(
        label="Refresh", options=["1m", "5m", "10m"], default_interval="5m"
    )
    refresh
    return (refresh,)


@app.cell
def _(optuna, os, refresh, study_name):
    refresh.value

    db_path = os.getenv("OPTUNA_DB_PATH")
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db_path}")

    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    df = df.rename(columns={"values_0": "flops", "values_1": "loss"})
    df["params_batch_size"] = 2 ** df["params_log2_batch_size"]
    df["params_hidden_dim"] = 2 ** df["params_log2_hidden_dim"]
    df["cpu_seconds"] = df["duration"].dt.total_seconds()
    df["steps/s"] = df["params_steps"] / df["cpu_seconds"]
    df
    return db_path, df, study


@app.cell
def _(curve_fit, df, mo, np):
    def pareto_frontier(df, cols, maximize=True):
        data = df[cols].values
        if not maximize:
            data = -data  # Flip to maximize if minimizing

        # Sort by first column (x)
        sorted_idx = np.argsort(data[:, 0])[::-1]  # Descending
        sorted_data = data[sorted_idx]

        pareto = [sorted_idx[0]]  # Always include first (best x)
        max_y = sorted_data[0, 1]  # Best y so far

        for idx in sorted_idx[1:]:
            if data[idx, 1] >= max_y:  # If y better or equal
                pareto.append(idx)
                max_y = data[idx, 1]

        return df.iloc[pareto].reset_index(drop=True)

    df_pareto = pareto_frontier(df, ["flops", "loss"], maximize=False)

    # Drop trials that were limited by the number of steps
    df_pareto = df_pareto[df_pareto["params_steps"] > 6000]
    df_pareto = df_pareto[df_pareto["params_steps"] < 90000]

    def L(flops, C_c, alpha_c, L_0):
        return C_c * flops**alpha_c + L_0

    popt, pconv = curve_fit(
        L,
        df_pareto["flops"],
        df_pareto["loss"],
        bounds=([0.0, -1.0, 0.0], [100, 0.0, 1.0]),
    )

    mo.vstack(
        [
            mo.md(f"$C_c = {popt[0]:.2f}$"),
            mo.md(f"$\\alpha_c = {popt[1]:.3f}$"),
            mo.md(f"$L_0 = {popt[2]:.2f}$"),
        ]
    )
    return L, df_pareto, pareto_frontier, pconv, popt


@app.cell
def _(L, alt, df_pareto, mo, np, pd, popt):
    # Plot points of the pareto front
    chart = (
        alt.Chart(df_pareto)
        .mark_point()
        .encode(
            x=alt.X("flops")
            .axis(grid=False, format="e")
            .scale(type="log", nice=False, padding=20),
            y=alt.Y("loss")
            .axis(grid=False, tickCount=10)
            .scale(nice=False, zero=False, padding=20),
            tooltip=[
                alt.Tooltip("flops", format=".1e"),
                alt.Tooltip("loss", format=".3f"),
                "number",
                "cpu_seconds",
                "steps/s",
                "params_steps",
                "params_batch_size",
                "params_hidden_layers",
                "params_hidden_dim",
                "params_learning_rate",
                "params_lr_cooldown_fraction",
            ],
        )
        .properties(width=500, height=500)
    )

    # Plot the regression
    flops = np.geomspace(df_pareto["flops"].min(), df_pareto["flops"].max())
    loss = L(flops, *popt)
    df_regression = pd.DataFrame({"flops": flops, "loss": loss})

    chart_regression = (
        alt.Chart(df_regression)
        .mark_line(color="black", strokeDash=[5, 5])
        .encode(x="flops", y="loss")
    )

    mo.hstack(
        [
            mo.ui.altair_chart(chart + chart_regression),
            mo.md(
                f"Loss after 1e17 flops: {L(1e17, *popt):.2f}<br>"
                f"Loss after 1e18 flops: {L(1e18, *popt):.2f}"
            ),
        ],
        widths=[2, 1],
    )
    return chart, chart_regression, df_regression, flops, loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Single Trial Explorer""")
    return


@app.cell
def _(df_pareto, mo):
    dropdown = mo.ui.dropdown(options=df_pareto["number"].astype(str))
    dropdown
    return (dropdown,)


@app.cell
def _(alt, dropdown, mo, study_name):
    from src.training.experiments.tune import read_trial_results

    mo.stop(dropdown.value is None)

    results = read_trial_results(study_name, int(dropdown.value))

    chart_loss = (
        alt.Chart(results)
        .mark_line(opacity=0.3)
        .encode(
            x="step",
            y=alt.Y("train_value_loss").scale(zero=False),
        )
        .properties(height=500)
    )

    chart_loss_ema = (
        alt.Chart(results)
        .mark_line()
        .encode(
            x="step",
            y=alt.Y("train_value_loss_ema").scale(zero=False),
        )
        .properties(height=500)
    )

    mo.ui.altair_chart(chart_loss + chart_loss_ema)
    return chart_loss, chart_loss_ema, read_trial_results, results


if __name__ == "__main__":
    app.run()
