import mlflow

if __name__ == "__main__":
    # mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(experiment_name="logisticregression")
    # mlflow.sklearn.log_model(
    #     "sklearn_model",
    #     "sklearn_model.pkl",
    #     registered_model_name="sklearn_model",
    #     signature=None,
    # )
    with mlflow.start_run(run_name="logisticregression"):
        mlflow.log_param("alpha", 0.5)
        mlflow.log_param("l1_ratio", 0.5)
        mlflow.log_metric("rmse", 0.5)
        mlflow.log_metric("r2", 0.5)
        mlflow.log_metric("mae", 0.5)

        mlflow.log_artifact("sklearn_model.pkl")
