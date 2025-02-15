import marimo

__generated_with = "0.11.5"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _(__file__):
    import marimo as mo
    from src.training.train import train
    from pathlib import Path
    from dotenv import load_dotenv
    import yaml
    import numpy as np
    import pandas as pd

    load_dotenv(Path(__file__).parents[3] / ".env")
    return Path, load_dotenv, mo, np, pd, train, yaml


@app.cell
def _(Path, __file__, mo, pd, train, yaml):
    #@mo.cache
    def train_experiment(config_name: str):
        with open(Path(__file__).parents[1] / f"configs/{config_name}.yaml") as f:
            config = yaml.safe_load(f)
        config["training"]["steps"] *= 2

        results = {}

        with mo.capture_stderr() as _stderr:
            with mo.capture_stdout() as _stdout:
                metrics = train(config)

        # Get the most recent lightning_log
        dir = Path(__file__).parents[1] / "lightning_logs"
        files = list(dir.glob("version_*"))
        version = max(int(f.stem.split("_")[-1]) for f in files)
        path = dir / f"version_{version}" / "metrics.csv"
        with open(path) as f:
            metrics = pd.read_csv(f)

        return metrics

    configs = [
        "debug",
        "pico",
    ]
    results = {config: train_experiment(config) for config in configs}
    return configs, results, train_experiment


@app.cell
def _(pd, results):
    import altair as alt

    for config, df in results.items():
        df['config'] = config

    df = pd.concat(results.values(), ignore_index=True)

    chart = alt.Chart(df).mark_line().encode(
        x='step',
        y=alt.Y('train_value_loss').scale(zero=False),
        color="config"
    )

    chart
    return alt, chart, config, df


if __name__ == "__main__":
    app.run()
