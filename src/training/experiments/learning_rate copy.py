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
def _(mo):
    run_button = mo.ui.run_button(label="Run experiment")
    run_button
    return (run_button,)


@app.cell
def _(CSVLogger, Path, __file__, mo, np, run_button, train, yaml):
    def train_experiment(config_name: str):
        with open(Path(__file__).parents[1] / f"configs/{config_name}.yaml") as f:
            config = yaml.safe_load(f)
        
        base_lr = config["model"]["learning_rate"]
        learning_rates = list(np.geomspace(0.1*base_lr, 10*base_lr, num=9))
        print(learning_rates)
        
        for i, lr in enumerate(mo.status.progress_bar(learning_rates)):
            config_copy = config.copy()
            print(type(lr))
            config_copy["model"]["learning_rate"] = lr

            csv_logger = CSVLogger(
                save_dir=Path(__file__).parents[1] / "experiment_logs",
                name="learning_rate",
                version=f"{config_name}_{i}",
            )
            
            with mo.capture_stderr() as _stderr:
                with mo.capture_stdout() as _stdout:
                    metrics = train(config_copy, csv_logger=csv_logger)

    if run_button.value:
        train_experiment("debug")
    return (train_experiment,)


@app.cell
def _(Path, __file__, mo, pd, yaml):
    import altair as alt

    config = "debug"
    results = []
    path = Path(__file__).parents[1] / "experiment_logs/learning_rate"

    for d in path.glob(f"{config}_*"):
        with open(d / "hparams.yaml") as f:
            hparams = yaml.safe_load(f)

        metrics = pd.read_csv(d / "metrics.csv")
        results.append({
            "learning_rate": hparams["learning_rate"],
            "loss": metrics.iloc[-1]["train_value_loss"]
        })

    df = pd.DataFrame(results)

    chart = alt.Chart(df).mark_point().encode(
        x=alt.X("learning_rate").scale(type="log"),
        y="loss",
    ).properties(
        width=500,
        height=500
    )

    #regression = chart.transform_regression("x", "y", method="quad").mark_line()

    mo.ui.altair_chart(chart)
    return alt, chart, config, d, df, f, hparams, metrics, path, results


if __name__ == "__main__":
    app.run()
