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
    import altair as alt
    import math

    load_dotenv(Path(__file__).parents[3] / ".env")
    return CSVLogger, Path, alt, load_dotenv, math, mo, np, pd, train, yaml


@app.cell
def _(mo):
    configs = {
        "debug": 3,
        "pico": 2,
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

        metrics = train(config, csv_logger=csv_logger)

    if run_button.value:
        for c, steps_mult in configs.items():
            if dictionnary.value[c]:
                train_experiment(c, steps_mult)
    return c, steps_mult, train_experiment


@app.cell
def _(mo, np):
    # eyeball fit a scaling law of the form L(C) = C_c / (C ^ alpha_c)
    x1 = 4e9
    y1 = mo.ui.slider(value=0.792, steps=np.linspace(0.6, 0.9, 301), full_width=True, label="y1")

    x2 = 3e10
    y2 = mo.ui.slider(value=0.769, steps=np.linspace(0.6, 0.9, 301), full_width=True, label="y2")
    return x1, x2, y1, y2


@app.cell
def _(Path, __file__, alt, configs, math, mo, np, pd, x1, x2, y1, y2):
    # Parsing results
    results = []
    for config in configs.keys():
        path = (
            Path(__file__).parents[1]
            / "experiment_logs/steps"
            / config
            / "metrics.csv"
        )

        if not path.exists():
            print(f"Warning: metrics for config '{config}' not found")
            continue

        df = pd.read_csv(path)
        df["config"] = config
        results.append(df)

    df = pd.concat(results, ignore_index=True)
    df["train_value_loss"] = df.groupby("config")["train_value_loss"].transform(
        lambda x: x.rolling(300, center=True, closed="both").mean()
    )
    df = df.dropna(subset=["train_value_loss"])


    # Calculating power law
    def power_function_params(x1, y1, x2, y2):
        n = (math.log(y2) - math.log(y1)) / (math.log(x2) - math.log(x1))
        a = y1 / (x1**n)
        return a, n


    C_c, alpha_c = power_function_params(
        x1,
        y1.value,
        x2,
        y2.value,
    )

    flops = np.geomspace(df["flops"].min(), df["flops"].max())
    loss = C_c * flops**alpha_c
    df_regression = pd.DataFrame({"flops": flops, "train_value_loss": loss})

    df_regression = df_regression[df["flops"].min() < df_regression["flops"]]
    df_regression = df_regression[df_regression["flops"] < df["flops"].max()]

    # Make the charts
    chart = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X("flops")
            .axis(grid=False, format="e")
            .scale(type="log", nice=False),
            y=alt.Y("train_value_loss")
            .axis(grid=False, values=np.linspace(0, 1, 51))
            .scale(type="log", nice=False),
            color="config",
            tooltip=[
                alt.Tooltip("flops", format=".1e"),
                alt.Tooltip("train_value_loss", format=".3f"),
                "step",
            ],
        )
        .properties(width=500, height=500)
    )

    chart_regression = (
        alt.Chart(df_regression)
        .mark_line(color="black", strokeDash=[5, 5])
        .encode(x="flops", y="train_value_loss")
    )

    df_points = pd.DataFrame(
        {"flops": [x1, x2], "train_value_loss": [y1.value, y2.value]}
    )
    chart_points = (
        alt.Chart(df_points)
        .mark_point(
            color="black",
        )
        .encode(x="flops", y="train_value_loss")
    )

    mo.vstack(
        [
            y1,
            y2,
            mo.md(f"y1={y1.value:.3f}"),
            mo.md(f"y2={y2.value:.3f}"),
            mo.md(f"C_c={C_c:.3f}"),
            mo.md(f"alpha_c={alpha_c:.3f}"),
            mo.ui.altair_chart(chart + chart_regression + chart_points),
        ]
    )
    return (
        C_c,
        alpha_c,
        chart,
        chart_points,
        chart_regression,
        config,
        df,
        df_points,
        df_regression,
        flops,
        loss,
        path,
        power_function_params,
        results,
    )


@app.cell
def _(C_c, alpha_c, mo):
    mo.md(
        f"Loss after 1e17 flops: {C_c * 1e17 ** alpha_c:.2f}<br>"
        f"Loss after 1e18 flops: {C_c * 1e18 ** alpha_c:.2f}"
    )
    return


if __name__ == "__main__":
    app.run()
