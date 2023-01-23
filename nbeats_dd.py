from darts.datasets import EnergyDataset
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from darts import TimeSeries
from darts.models.forecasting.nbeats import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score

from darts.datasets import EnergyDataset
def display_forecast(pred_series, ts_transformed, forecast_type, start_date=None):
    plt.figure(figsize=(8, 5))
    if start_date:
        ts_transformed = ts_transformed.drop_before(start_date)
    ts_transformed.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    plt.title(
        "R2: {}".format(r2_score(ts_transformed.univariate_component(0), pred_series))
    )
    plt.legend()
    return r2_score(ts_transformed.univariate_component(0), pred_series)

if __name__ == '__main__':
    results = []
    for i in range(20):
        df = EnergyDataset().load().pd_dataframe()
        df_day_avg = df.groupby(df.index.astype(str).str.split(" ").str[0]).mean().reset_index()
        filler = MissingValuesFiller()
        scaler = Scaler()
        series = scaler.fit_transform(
            filler.transform(
                TimeSeries.from_dataframe(
                    df_day_avg, "time", ["generation hydro run-of-river and poundage"]
                )
            )
        ).astype(np.float32)
        train, val = series.split_after(pd.Timestamp("20170901"))
        model_nbeats = NBEATSModel(
            input_chunk_length=30,
            output_chunk_length=7,
            generic_architecture=True,
            num_stacks=3,
            num_blocks=3,
            num_layers=4,
            layer_widths=128,
            n_epochs=50,
            nr_epochs_val_period=1,
            batch_size=800,
            model_name="nbeats_run",
        )
        model_nbeats.fit(train, val_series=val, verbose=True)

        model_nbeats.set_online(True)
        model_nbeats.n_epochs = 10
        pred_series = model_nbeats.historical_forecasts(
            series,
            start=pd.Timestamp("20170901"),
            forecast_horizon=7,
            stride=5,
            retrain=False,
            last_points_only=False,
            verbose=True,

        )
        print("Finished")
        results.append(display_forecast(pred_series, series, "7 day", start_date=pd.Timestamp("20170901")))
        plt.show()
    print(results, np.mean(results), np.var(results))