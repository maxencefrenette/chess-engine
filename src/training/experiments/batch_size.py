import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _(__file__):
    import marimo as mo
    from src.training.train import train, load_config
    from pathlib import Path
    from dotenv import load_dotenv
    import yaml
    import numpy as np
    import pandas as pd
    from lightning.pytorch.loggers import CSVLogger
    import altair as alt
    import math
    import lightning as L
    from src.training.experiments.utils import read_experiment_results, smooth_column

    load_dotenv(Path(__file__).parents[3] / ".env")
    return (
        CSVLogger,
        L,
        Path,
        alt,
        load_config,
        load_dotenv,
        math,
        mo,
        np,
        pd,
        read_experiment_results,
        smooth_column,
        train,
        yaml,
    )


@app.cell
def _(mo):
    configs = ["debug", "pico"]

    checkboxes = {config: mo.ui.checkbox() for config in configs}
    dictionnary = mo.ui.dictionary(checkboxes)
    run_button = mo.ui.run_button(label="Run experiments")

    mo.vstack([dictionnary, run_button])
    return checkboxes, configs, dictionnary, run_button


@app.cell
def _(
    CSVLogger,
    L,
    Path,
    __file__,
    configs,
    dictionnary,
    load_config,
    run_button,
    train,
):
    class GradientLoggingCallback(L.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            total_norm = 0.0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            # Log the aggregated gradient norm for the current step
            pl_module.log(f"grad_norm_{batch_idx % 4}", total_norm, on_step=True, on_epoch=False)


    def train_experiment(config_name: str):
        config = load_config(config_name)
        config["steps"] = 2000
        config["batch_size"] = 4

        csv_logger = CSVLogger(
            save_dir=Path(__file__).parents[1] / "experiment_logs",
            name="batch_size",
            version=config_name,
        )

        metrics = train(
            config,
            csv_logger=csv_logger,
            accumulate_grad_batches=4,
            extra_callbacks=[GradientLoggingCallback()]
        )

    if run_button.value:
        for c in configs:
            if dictionnary.value[c]:
                train_experiment(c)
    return GradientLoggingCallback, c, train_experiment


@app.cell
def _(alt, mo, read_experiment_results):
    df = read_experiment_results("batch_size")
    df["b_small"] = df["batch_size"]
    df["b_big"] = 4 * df["batch_size"]

    smoothing_window_size = 300
    for i in range(4):
        df = smooth_column(df, f"grad_norm_{i}", window_size=smoothing_window_size)

    df["|G_B_small|"] = df["grad_norm_0"]
    df["|G_B_big|"] = df["grad_norm_3"] / 4
    df["|G|^2"] = (
        1 / (df["b_big"] - df["b_small"])
        * (df["b_big"] * df["|G_B_big|"]**2 - df["b_small"] * df["|G_B_small|"]**2)
    )
    df["S"] = 1 / (1/df["b_small"] - 1/df["b_big"]) * (df["|G_B_small|"]**2 - df["|G_B_big|"]**2)

    # smooth out values
    df = smooth_column(df, "|G|^2", window_size=100)
    df = smooth_column(df, "S", window_size=100)

    df["b_simple"] = df["S"] / df["|G|^2"]

    # Make the charts
    chart_b_small = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("step")
                .axis(grid=False),
            y=alt.Y("grad_norm_0")
                .axis(grid=False),
            color="config",
        )
        .properties(width=250, height=250)
    )

    chart_b_big = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("step")
                .axis(grid=False),
            y=alt.Y("grad_norm_3")
                .axis(grid=False),
            color="config",
        )
        .properties(width=250, height=250)
    )

    chart_g_squared = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("step")
                .axis(grid=False),
            y=alt.Y("|G|^2")
                .axis(grid=False),
            color="config",
        )
        .properties(width=250, height=250)
    )

    chart_s = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("step")
                .axis(grid=False),
            y=alt.Y("S")
                .axis(grid=False),
            color="config",
        )
        .properties(width=250, height=250)
    )

    chart_b_simple = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("step")
                .axis(grid=False),
            y=alt.Y("b_simple")
                .axis(grid=False),
            color="config",
        )
        .properties(width=500, height=500)
    )

    mo.vstack([
        mo.hstack([chart_b_small, chart_b_big]),
        mo.hstack([chart_g_squared, chart_s]),
        chart_b_simple,
    ])
    return (
        chart_b_big,
        chart_b_simple,
        chart_b_small,
        chart_g_squared,
        chart_s,
        df,
        i,
        smoothing_window_size,
    )


if __name__ == "__main__":
    app.run()
