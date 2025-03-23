import os
import pickle
import click
import mlflow
import numpy as np  # ✅ Добавлено

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run() as run:
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # ✅ Ручной расчёт RMSE
        val_preds = rf.predict(X_val)
        test_preds = rf.predict(X_test)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Log model explicitly
        mlflow.sklearn.log_model(rf, artifact_path="model")

        return run.info.run_id, test_rmse


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Step 1: Get top-N HPO runs
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    # Step 2: Re-train and evaluate top-N runs
    results = []
    for run in hpo_runs:
        run_id, test_rmse = train_and_log_model(data_path=data_path, params=run.data.params)
        results.append((run_id, test_rmse))

    # Step 3: Pick the best one based on test RMSE
    best_run_id, best_rmse = sorted(results, key=lambda x: x[1])[0]

    # Step 4: Register the best model
    model_uri = f"runs:/{best_run_id}/model"
    model_name = "random-forest-best-model"

    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Registered model from run {best_run_id} with test RMSE: {best_rmse:.4f}")


if __name__ == '__main__':
    run_register_model()
