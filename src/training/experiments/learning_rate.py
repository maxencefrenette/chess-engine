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

    load_dotenv(Path(__file__).parents[3] / ".env")
    return CSVLogger, Path, load_dotenv, mo, np, pd, train, yaml


@app.cell
def _(mo, np):
    configs = {
        "debug": np.geomspace(0.00001, 0.1, num=30).tolist(),
        "pico": np.geomspace(0.00001, 0.1, num=30).tolist(),
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
    run_button,
    train,
    yaml,
):
    def train_experiment(config_name: str, learning_rates: list[float]):
        with open(Path(__file__).parents[1] / f"configs/{config_name}.yaml") as f:
            config = yaml.safe_load(f)

        base_lr = config["learning_rate"]

        for i, lr in enumerate(mo.status.progress_bar(learning_rates)):
            config_copy = config.copy()
            config_copy["learning_rate"] = lr

            csv_logger = CSVLogger(
                save_dir=Path(__file__).parents[1] / "experiment_logs",
                name="learning_rate",
                version=f"{config_name}_{i}",
            )

            with mo.capture_stderr() as _stderr:
                with mo.capture_stdout() as _stdout:
                    metrics = train(config_copy, csv_logger=csv_logger)

    if run_button.value:
        for c, learning_rates in configs.items():
            if dictionnary.value[c]:
                train_experiment(c, learning_rates)
    return c, learning_rates, train_experiment


@app.cell
def _(Path, __file__, configs, mo, pd, yaml):
    import altair as alt

    config = "debug"
    path = Path(__file__).parents[1] / "experiment_logs/learning_rate"
    results = []
    training_logs = []

    for config in configs.keys():
        for d in path.glob(f"{config}_*"):
            with open(d / "hparams.yaml") as f:
                hparams = yaml.safe_load(f)
        
            metrics = pd.read_csv(d / "metrics.csv")
            results.append({
                "config": config,
                "learning_rate": hparams["learning_rate"],
                "loss": metrics.iloc[-50]["train_value_loss"].mean()
            })
        
            metrics["config"] = config
            metrics["learning_rate"] = hparams["learning_rate"]
            training_logs.append(metrics)

    df = pd.DataFrame(results)

    chart = alt.Chart(df).mark_point().encode(
        x=alt.X("learning_rate").scale(type="log", nice=False),
        y=alt.Y("loss").scale(zero=False, padding=20),
        color="config"
    ).properties(
        height=500
    )

    loess = chart.transform_loess(
        "learning_rate",
        "loss",
        groupby=["config"],
        bandwidth=0.3
    ).mark_line()

    mo.ui.altair_chart(chart + loess)
    return (
        alt,
        chart,
        config,
        d,
        df,
        f,
        hparams,
        loess,
        metrics,
        path,
        results,
        training_logs,
    )


@app.cell
def _(alt, mo, pd, training_logs):
    df2 = pd.concat(training_logs, ignore_index=True)
    df2["train_value_loss"] = df2.groupby(["config", "learning_rate"])["train_value_loss"] \
        .transform(lambda x: x.rolling(50, center=True, closed="both").mean())
    df2 = df2.dropna(subset=["train_value_loss"])

    def make_loss_chart(df, config_name: str):
        return alt.Chart(df[df["config"] == config_name]).mark_line().encode(
            x=alt.X("step").scale(zero=False, nice=False),
            y=alt.Y("train_value_loss").scale(zero=False),
            color=alt.Color("learning_rate:Q").scale(type="log", scheme="viridis"),
            tooltip=["learning_rate"]
        ).properties(
            height=500
        )

    mo.vstack([
        mo.ui.altair_chart(make_loss_chart(df2, "debug")),
        mo.ui.altair_chart(make_loss_chart(df2, "pico")),
    ])
    return df2, make_loss_chart


if __name__ == "__main__":
    app.run()
