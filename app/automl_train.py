"""AutoML training utilities for Vertex AI time-series forecasting."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import AutoMLForecastingTrainingJob

# Load environment variables from .env file
load_dotenv()


def upload_data_to_gcs(local_file_path: str, gcs_path: str, project_id: str) -> str:
    """Upload local data file to Google Cloud Storage."""
    # Parse GCS path
    if not gcs_path.startswith("gs://"):
        raise ValueError("GCS path must start with gs://")

    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name = path_parts[1] if len(path_parts) > 1 else ""

    try:
        # Initialize storage client
        storage_client = storage.Client(project=project_id)
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Upload file
        print(f"Uploading {local_file_path} to {gcs_path}")
        blob.upload_from_filename(local_file_path)
        print(f"✅ Data uploaded successfully to {gcs_path}")

        return gcs_path
    except Exception as e:
        print(f"❌ Failed to upload data to GCS: {str(e)}")
        raise


def create_dataset_and_train_automl(
    project_id: str,
    region: str,
    dataset_path: str,
    model_display_name: str = "thrasio-automl-forecast-model",
    target_column: str = "revenue",
    time_column: str = "date",
) -> str:
    """Create dataset and train AutoML time-series forecasting model."""

    # Initialize Vertex AI
    staging_bucket = f"gs://{project_id}-vertex-pipelines-pratik-20250903"
    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket)

    print(f"Creating AutoML forecasting dataset from: {dataset_path}")

    # Create tabular dataset for time-series forecasting
    dataset = aiplatform.TimeSeriesDataset.create(
        display_name=f"{model_display_name}-dataset",
        gcs_source=[dataset_path],
    )

    print(f"Dataset created: {dataset.resource_name}")

    # Create AutoML forecasting training job
    print("Starting AutoML training for time-series forecasting...")
    print(f"Target column: {target_column}")
    print(f"Time column: {time_column}")

    job = AutoMLForecastingTrainingJob(
        display_name=f"{model_display_name}-training-job",
        optimization_objective="minimize-rmse",
        column_transformations=[
            {"timestamp": {"column_name": "date"}},
            {"numeric": {"column_name": "revenue"}},
            {"numeric": {"column_name": "search_volume"}},
            {"numeric": {"column_name": "click_through_rate"}},
            {"numeric": {"column_name": "conversion_rate"}},
            {"categorical": {"column_name": "day_of_week"}},
            {"categorical": {"column_name": "month"}},
            {"categorical": {"column_name": "quarter"}},
        ],
    )

    # Configure forecasting parameters
    model = job.run(
        dataset=dataset,
        target_column=target_column,
        time_column=time_column,
        time_series_identifier_column="category",  # Group forecasts by category
        unavailable_at_forecast_columns=[
            target_column
        ],  # Target column not available at forecast time
        available_at_forecast_columns=[
            "date",
            "search_volume",
            "click_through_rate",
            "conversion_rate",
            "day_of_week",
            "month",
            "quarter",
        ],  # Features available at forecast time
        forecast_horizon=30,  # Predict 30 days ahead
        context_window=365,  # Use 1 year of historical data
        data_granularity_unit="day",
        data_granularity_count=1,
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        budget_milli_node_hours=1000,  # 1 node hour for quick training
    )

    print(f"✅ AutoML model training completed: {model.resource_name}")
    return model.resource_name


def evaluate_model_performance(model_resource_name: str):
    """Evaluate the trained AutoML model and get performance metrics."""
    try:
        # Get the model
        model = aiplatform.Model(model_resource_name)

        print(f"Model: {model.display_name}")
        print("Model Type: Time-series Forecasting")
        print("Training Algorithm: AutoML")

        # Get model evaluation metrics
        evaluations = model.list_model_evaluations()

        if evaluations:
            eval_metrics = evaluations[0]
            metrics = eval_metrics.metrics

            print("\n📊 Performance Metrics:")

            # Common time-series metrics
            if "rootMeanSquaredError" in metrics:
                print(f"RMSE: {metrics['rootMeanSquaredError']:.4f}")
            if "meanAbsoluteError" in metrics:
                print(f"MAE: {metrics['meanAbsoluteError']:.4f}")
            if "meanAbsolutePercentageError" in metrics:
                print(f"MAPE: {metrics['meanAbsolutePercentageError']:.4f}%")
            if "rSquared" in metrics:
                print(f"R²: {metrics['rSquared']:.4f}")

            # Additional forecasting-specific metrics
            for key, value in metrics.items():
                if key not in [
                    "rootMeanSquaredError",
                    "meanAbsoluteError",
                    "meanAbsolutePercentageError",
                    "rSquared",
                ]:
                    print(f"{key}: {value}")
        else:
            print("No evaluation metrics available yet.")

    except Exception as e:
        print(f"Error evaluating model: {str(e)}")


def main():
    """Main AutoML training script."""

    # Get environment variables
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "us-central1")

    if not project_id:
        print("Error: GCP_PROJECT_ID environment variable not set")
        sys.exit(1)

    print(f"Training AutoML model in project: {project_id}, region: {region}")

    try:
        # First generate sample data
        sys.path.insert(0, str(Path(__file__).parent))
        from sample_data import save_sample_data

        # Generate and save sample data
        full_path, daily_path = save_sample_data()

        # Upload data to GCS for AutoML
        staging_bucket = f"gs://{project_id}-vertex-pipelines-pratik-20250903"
        gcs_data_path = f"{staging_bucket}/data/amazon_search_daily.csv"

        # Upload local data to GCS
        upload_data_to_gcs(daily_path, gcs_data_path, project_id)

        # Train AutoML model
        model_resource_name = create_dataset_and_train_automl(
            project_id=project_id,
            region=region,
            dataset_path=gcs_data_path,
            target_column="revenue",
            time_column="date",
        )

        # Evaluate model performance
        print("\n🔍 Evaluating model performance...")
        evaluate_model_performance(model_resource_name)

        print("\n✅ AutoML baseline model training completed!")
        print(f"Model resource: {model_resource_name}")

    except Exception as e:
        print(f"❌ AutoML training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
