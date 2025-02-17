import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    from src.training.train import train
    from pathlib import Path
    from dotenv import load_dotenv
    import yaml
    import numpy as np

    load_dotenv(Path(__file__).parents[2] / ".env")
    return Path, load_dotenv, mo, np, train, yaml


@app.cell
def _(Path, __file__, mo, np, train, yaml):
    config = "pico"
    learning_rates = np.geomspace(0.005, 0.1, num=10)

    with open(Path(__file__).parent / f"configs/{config}.yaml") as f:
        config = yaml.safe_load(f)["model"]

    results = {}
    for l in mo.status.progress_bar(learning_rates):
        config_copy = config.copy()
        config_copy["learning_rate"] = l

        with mo.capture_stderr() as _stderr:
            with mo.capture_stdout() as _stdout:
                metrics = train(config_copy)

        results[l] = metrics
    return config, config_copy, f, l, learning_rates, metrics, results


@app.cell
def _(results):
    import pandas as pd
    import altair as alt

    df = pd.DataFrame.from_dict(results, orient="index")
    df = df.reset_index().rename(columns={"index": "learning_rate"})
    df["train_value_loss"] = df["train_value_loss"].map(lambda x: x.item())
    df["train_value_accuracy"] = df["train_value_accuracy"].map(lambda x: x.item())

    chart = alt.Chart(df).mark_point().encode(
        x=alt.X('learning_rate').scale(type='log'),
        y='train_value_loss',
    )

    # mo.ui.altair_chart(chart)
    chart
    return alt, chart, df, pd


if __name__ == "__main__":
    app.run()
