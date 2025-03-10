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
    return Path, alt, curve_fit, load_dotenv, mo, np, optuna, os, pd


@app.cell
def _(optuna, os):
    db_path = os.getenv("OPTUNA_DB_PATH")
    study = optuna.load_study(study_name="tune_v3", storage=f"sqlite:///{db_path}")

    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    df = df.rename(columns={"values_0": "flops", "values_1": "loss"})
    df["params_hidden_dim"] = 2 ** df["params_log2_hidden_dim"]
    df["cpu_seconds"] = df["duration"].dt.total_seconds()
    df["steps/s"] = df["params_steps"] / df["cpu_seconds"]
    df
    return db_path, df, study


@app.cell
def _(curve_fit, df, np):
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

    # Drop outliers from the start and end of the pareto frontier
    df_pareto = df_pareto.iloc[1:].reset_index(drop=True)

    def L(flops, C_c, alpha_c):
        return C_c * flops**alpha_c

    popt, pconv = curve_fit(
        L, df_pareto["flops"], df_pareto["loss"], bounds=([0.0, -1.0], [100, 0.0])
    )
    popt
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
            .axis(grid=False, values=np.linspace(0, 1, 51))
            .scale(type="log", nice=False, padding=20),
            tooltip=[
                alt.Tooltip("flops", format=".1e"),
                alt.Tooltip("loss", format=".3f"),
                "params_steps",
                "params_hidden_layers",
                "params_hidden_dim",
                "params_lr_cooldown_fraction",
                "steps/s",
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

    mo.ui.altair_chart(chart + chart_regression)
    return chart, chart_regression, df_regression, flops, loss


@app.cell
def _(L, mo, popt):
    mo.md(
        f"Loss after 1e17 flops: {L(1e17, *popt):.2f}<br>"
        f"Loss after 1e18 flops: {L(1e18, *popt):.2f}"
    )
    return


if __name__ == "__main__":
    app.run()
