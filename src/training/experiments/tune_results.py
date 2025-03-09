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
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parents[3] / ".env")
    return Path, alt, load_dotenv, mo, np, optuna, os


@app.cell
def _(optuna, os):
    db_path = os.getenv("OPTUNA_DB_PATH")
    study = optuna.load_study(study_name="tune_v2", storage=f"sqlite:///{db_path}")

    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"]
    df = df.rename(columns={"values_0": "flops", "values_1": "loss"})
    df["params_hidden_dim"] = 2 ** df["params_log2_hidden_dim"]
    df
    return db_path, df, study


@app.cell
def _(alt, df, mo, np):
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

    # Drop first element because it doesn't follow the trend
    df_pareto = df_pareto.iloc[2:].reset_index(drop=True)

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
            ],
        )
        .properties(width=500, height=500)
    )

    mo.ui.altair_chart(chart)
    return chart, df_pareto, pareto_frontier


if __name__ == "__main__":
    app.run()
