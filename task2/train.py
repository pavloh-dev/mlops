import os
import pickle
import click
from math import sqrt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    # ✅ Set experiment name
    mlflow.set_tracking_uri("file:///./mlruns")

    mlflow.set_experiment("random-forest-baseline")

    # ✅ Enable autologging
    mlflow.sklearn.autolog()

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

    with mlflow.start_run(run_name="rf-baseline"):
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = sqrt(mean_squared_error(y_val, y_pred))
        print(f"RMSE: {rmse:.4f}")
        print(f"min_samples_split: {rf.min_samples_split}")

        # ✅ Explicitly log RMSE and params (in case autolog fails)
        mlflow.log_metric("rmse_manual", rmse)
        mlflow.log_param("min_samples_split", rf.min_samples_split)


if __name__ == '__main__':
    run_train()
