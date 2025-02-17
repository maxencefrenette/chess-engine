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
    import pandas as pd
    from lightning.pytorch.loggers import CSVLogger

    load_dotenv(Path(__file__).parents[2] / ".env")
    return CSVLogger, Path, load_dotenv, mo, np, pd, train, yaml


@app.cell
def _(mo):
    configs = {
        "debug": 10,
        "pico": 3,
    }

    checkboxes = {config: mo.ui.checkbox() for config in configs.keys()}
    dictionnary = mo.ui.dictionary(checkboxes)
    run_button = mo.ui.run_button(label="Run experiments")

    mo.vstack([dictionnary, run_button])
    return checkboxes, configs, dictionnary, run_button


@app.cell
def _(
    CSVLogger,
    Path,
    __file__,
    configs,
    dictionnary,
    mo,
    pd,
    run_button,
    train,
    yaml,
):
    def train_experiment(config_name: str, steps_mult):
        with open(Path(__file__).parents[1] / f"configs/{config_name}.yaml") as f:
            config = yaml.safe_load(f)
        config["training"]["steps"] *= steps_mult

        csv_logger = CSVLogger(
            save_dir=Path(__file__).parents[1] / "experiment_logs",
            name="steps",
            version=config_name,
        )

        with mo.capture_stderr() as _stderr:
            with mo.capture_stdout() as _stdout:
                metrics = train(config, csv_logger=csv_logger)

        # Get the most recent lightning_log
        dir = Path(__file__).parents[1] / "lightning_logs"
        files = list(dir.glob("version_*"))
        version = max(int(f.stem.split("_")[-1]) for f in files)
        path = dir / f"version_{version}" / "metrics.csv"
        with open(path) as f:
            metrics = pd.read_csv(f)

    if run_button.value:
        for c, steps_mult in configs.items():
            if dictionnary.value[c]:
                train_experiment(c, steps_mult)
    return c, steps_mult, train_experiment


@app.cell
def _(Path, __file__, configs, np, pd):
    import altair as alt

    # eyeball fit a scaling law of the form L(C) = C_c / (C ^ alpha_c)
    C_c = 0
    alpha_c = 0

    results = []
    for config in configs.keys():
        path = Path(__file__).parents[1] / "experiment_logs/steps" / config / "metrics.csv"

        if not path.exists():
            print(f"Warning: metrics for config '{config}' not found")
            continue

        df = pd.read_csv(path)
        df["config"] = config
        results.append(df)

    df = pd.concat(results, ignore_index=True)
    df['train_value_loss'] = df.groupby('config')['train_value_loss'] \
        .transform(lambda x: x.rolling(100, center=True, closed="both").mean())
    df = df.dropna(subset=["train_value_loss"])
    df = df[df["train_value_loss"] < 0.9]


    chart = alt.Chart(df).mark_line().encode(
        x=alt.X("flops") \
            .axis(grid=False, format="e") \
            .scale(type="log", nice=False),
        y=alt.Y("train_value_loss") \
            .axis(grid=False, values=np.linspace(0, 1, 51)) \
            .scale(type="log", nice=False),
        color="config"
    ).properties(
        width=500,
        height=500
    )

    chart
    return C_c, alpha_c, alt, chart, config, df, path, results


if __name__ == "__main__":
    app.run()
