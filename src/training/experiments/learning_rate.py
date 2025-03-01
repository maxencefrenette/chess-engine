import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    import yaml
    from dotenv import load_dotenv
    from lightning.pytorch.loggers import CSVLogger

    from src.training.experiments.utils import read_experiment_results, smooth_column
    from src.training.train import load_config, train

    load_dotenv(Path(__file__).parents[3] / ".env")
    return (
        CSVLogger,
        Path,
        load_config,
        load_dotenv,
        mo,
        np,
        pd,
        read_experiment_results,
        smooth_column,
        train,
        yaml,
    )


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
    load_config,
    mo,
    run_button,
    train,
):
    def train_experiment(config_name: str, learning_rates: list[float]):
        config = load_config(config_name)

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
def _(mo, read_experiment_results):
    import altair as alt

    # Read all experiment results
    df = read_experiment_results("learning_rate")

    # Group by config and learning_rate, calculate mean of last 50 steps or all available
    df_learning_rates = (
        df.groupby(["config", "learning_rate"]).agg(
            {"train_value_loss": lambda x: x.tail(50).mean()}
        )
        # .apply(lambda x: x.tail(50)["train_value_loss"].mean())
        .reset_index()
    )

    chart = (
        alt.Chart(df_learning_rates)
        .mark_point()
        .encode(
            x=alt.X("learning_rate").scale(type="log", nice=False),
            y=alt.Y("train_value_loss").scale(zero=False, domainMax=1.05, padding=20),
            color="config",
        )
        .properties(height=500)
    )

    loess = chart.transform_loess(
        "learning_rate", "train_value_loss", groupby=["config"], bandwidth=0.3
    ).mark_line()

    mo.ui.altair_chart(chart + loess)
    return alt, chart, df, df_learning_rates, loess


@app.cell
def _(alt, df, mo, smooth_column):
    df2 = smooth_column(
        df, "train_value_loss", window_size=50, group_by=["config", "learning_rate"]
    )

    def make_loss_chart(df, config_name: str):
        return (
            alt.Chart(df[df["config"] == config_name])
            .mark_line()
            .encode(
                x=alt.X("step").scale(zero=False, nice=False),
                y=alt.Y("train_value_loss").scale(zero=False, domain=(0.7, 1.15)),
                color=alt.Color("learning_rate:Q").scale(type="log", scheme="viridis"),
                tooltip=["learning_rate"],
            )
            .properties(height=500)
        )

    mo.vstack(
        [
            mo.ui.altair_chart(make_loss_chart(df2, "debug")),
            mo.ui.altair_chart(make_loss_chart(df2, "pico")),
        ]
    )
    return df2, make_loss_chart


if __name__ == "__main__":
    app.run()
