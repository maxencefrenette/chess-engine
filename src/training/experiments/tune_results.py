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
    study = optuna.load_study(study_name="tune", storage=f"sqlite:///{db_path}")

    df = study.trials_dataframe()
    df = df.rename(columns={"values_0": "flops", "values_1": "loss"})
    df
    return db_path, df, study


@app.cell
def _(alt, df, mo, np):
    chart = (
        alt.Chart(df)
        .mark_point()
        .encode(
            x=alt.X("flops").axis(grid=False, format="e").scale(type="log", nice=False),
            y=alt.Y("loss")
            .axis(grid=False, values=np.linspace(0, 1, 51))
            .scale(type="log", nice=False),
            tooltip=[
                alt.Tooltip("flops", format=".1e"),
                alt.Tooltip("loss", format=".3f"),
            ],
        )
        .properties(width=500, height=500)
    )

    mo.ui.altair_chart(chart)
    return (chart,)


if __name__ == "__main__":
    app.run()
