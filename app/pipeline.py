"""Vertex AI Pipeline Definition for Thrasio ML MVP."""

from typing import NamedTuple

from google.cloud import aiplatform
from kfp.dsl import component, pipeline


@component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-bigquery", "pandas", "numpy", "pydantic"],
)
def data_ingestion_component(
    project_id: str,
    bq_table_name: str,
    region: str = "us-central1",
    validation_config: dict = None,
) -> NamedTuple(
    "Outputs",
    [
        ("ingested_data_path", str),
        ("validation_results", dict),
        ("row_count", int),
        ("schema_info", dict),
    ],
):
    """Ingest data from BigQuery and perform basic validation checks."""
    from collections import namedtuple
    from typing import Any, Dict

    import pandas as pd
    from google.cloud import bigquery

    def validate_dataframe(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform basic data validation checks."""
        validation_results = {
            "passed": True,
            "checks": {},
            "errors": [],
            "warnings": [],
        }

        # Basic checks
        validation_results["checks"]["row_count"] = len(df)
        validation_results["checks"]["column_count"] = len(df.columns)
        validation_results["checks"]["null_counts"] = df.isnull().sum().to_dict()
        validation_results["checks"]["data_types"] = df.dtypes.astype(str).to_dict()

        # Minimum row count validation
        min_rows = config.get("min_rows", 1) if config else 1
        if len(df) < min_rows:
            validation_results["passed"] = False
            validation_results["errors"].append(
                f"Row count {len(df)} is below minimum threshold {min_rows}"
            )

        # Required columns validation
        required_cols = config.get("required_columns", []) if config else []
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            validation_results["passed"] = False
            validation_results["errors"].append(
                f"Missing required columns: {list(missing_cols)}"
            )

        # Null value thresholds
        null_thresholds = config.get("null_thresholds", {}) if config else {}
        for col, threshold in null_thresholds.items():
            if col in df.columns:
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                if null_pct > threshold:
                    validation_results["warnings"].append(
                        f"Column {col} has {null_pct:.1f}% null values, "
                        f"exceeding threshold {threshold}%"
                    )

        return validation_results

    try:
        print(f"Starting data ingestion from BigQuery table: {bq_table_name}")

        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)

        # Construct query with table name parameter
        query = f"""
        SELECT *
        FROM `{bq_table_name}`
        LIMIT 10000
        """

        print(f"Executing query: {query}")

        # Execute query and load to DataFrame
        query_job = client.query(query)
        df = query_job.to_dataframe()

        print(f"Successfully loaded {len(df)} rows from BigQuery")

        # Perform data validation
        validation_config_dict = validation_config or {}
        validation_results = validate_dataframe(df, validation_config_dict)

        # Store data to GCS (simulated path for now)
        table_name_clean = bq_table_name.replace(".", "_")
        output_path = (
            f"gs://{project_id}-ml-data/ingested/{table_name_clean}/data.parquet"
        )

        # In a real implementation, you would save to GCS here
        print(f"Data would be saved to: {output_path}")

        # Extract schema information
        schema_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "shape": df.shape,
            "memory_usage": df.memory_usage(deep=True).sum(),
        }

        print("Data ingestion and validation completed successfully")
        print(f"Validation passed: {validation_results['passed']}")

        if validation_results["errors"]:
            print(f"Validation errors: {validation_results['errors']}")
        if validation_results["warnings"]:
            print(f"Validation warnings: {validation_results['warnings']}")

        Outputs = namedtuple(
            "Outputs",
            ["ingested_data_path", "validation_results", "row_count", "schema_info"],
        )
        return Outputs(output_path, validation_results, len(df), schema_info)

    except Exception as e:
        print(f"Error during data ingestion: {str(e)}")
        error_results = {
            "passed": False,
            "checks": {},
            "errors": [f"Data ingestion failed: {str(e)}"],
            "warnings": [],
        }

        Outputs = namedtuple(
            "Outputs",
            ["ingested_data_path", "validation_results", "row_count", "schema_info"],
        )
        return Outputs("", error_results, 0, {})


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "pandas",
        "numpy",
        "scikit-learn",
        "google-cloud-storage",
        "pyarrow",
    ],
)
def data_preprocessing_component(
    project_id: str,
    region: str,
    input_dataset: str,
    preprocessing_config: dict = None,
    timestamp_column: str = "timestamp",
    target_column: str = None,
) -> NamedTuple(
    "Outputs",
    [
        ("processed_data_path", str),
        ("metadata", dict),
        ("feature_names", list),
        ("scaler_info", dict),
    ],
):
    """Time-series specific preprocessing component for ML training."""
    from collections import namedtuple
    from datetime import datetime
    from typing import List

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

    def create_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create time-based features from timestamp column."""
        df = df.copy()

        # Ensure timestamp column is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        # Extract time components
        df["year"] = df[timestamp_col].dt.year
        df["month"] = df[timestamp_col].dt.month
        df["day"] = df[timestamp_col].dt.day
        df["hour"] = df[timestamp_col].dt.hour
        df["minute"] = df[timestamp_col].dt.minute
        df["day_of_week"] = df[timestamp_col].dt.dayofweek
        df["day_of_year"] = df[timestamp_col].dt.dayofyear
        df["week_of_year"] = df[timestamp_col].dt.isocalendar().week
        df["quarter"] = df[timestamp_col].dt.quarter

        # Cyclical features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        print(
            "Created time features: year, month, day, hour, minute, day_of_week, "
            "day_of_year, week_of_year, quarter, cyclical features"
        )

        return df

    def create_lag_features(
        df: pd.DataFrame, target_col: str, lags: List[int]
    ) -> pd.DataFrame:
        """Create lag features for time-series forecasting."""
        if target_col not in df.columns:
            print(f"Target column {target_col} not found, skipping lag features")
            return df

        df = df.copy()

        for lag in lags:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        lag_feature_names = [f"{target_col}_lag_{lag}" for lag in lags]
        print(f"Created lag features for {target_col}: {lag_feature_names}")
        return df

    def create_rolling_features(
        df: pd.DataFrame, target_col: str, windows: List[int]
    ) -> pd.DataFrame:
        """Create rolling window features."""
        if target_col not in df.columns:
            print(f"Target column {target_col} not found, skipping rolling features")
            return df

        df = df.copy()

        for window in windows:
            df[f"{target_col}_rolling_mean_{window}"] = (
                df[target_col].rolling(window=window).mean()
            )
            df[f"{target_col}_rolling_std_{window}"] = (
                df[target_col].rolling(window=window).std()
            )
            df[f"{target_col}_rolling_min_{window}"] = (
                df[target_col].rolling(window=window).min()
            )
            df[f"{target_col}_rolling_max_{window}"] = (
                df[target_col].rolling(window=window).max()
            )

        print(f"Created rolling features for {target_col} with windows: {windows}")
        return df

    def encode_categorical_features(
        df: pd.DataFrame, categorical_cols: List[str]
    ) -> tuple:
        """Encode categorical features using label encoding."""
        df = df.copy()
        encoders = {}

        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = {
                    "type": "label_encoder",
                    "classes": le.classes_.tolist(),
                }
                print(f"Encoded categorical column: {col}")

        return df, encoders

    def scale_numerical_features(
        df: pd.DataFrame, numerical_cols: List[str], scaler_type: str = "standard"
    ) -> tuple:
        """Scale numerical features."""
        df = df.copy()
        scalers = {}

        available_cols = [col for col in numerical_cols if col in df.columns]

        if not available_cols:
            print("No numerical columns found for scaling")
            return df, scalers

        if scaler_type == "standard":
            scaler = StandardScaler()
        elif scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            print(f"Unknown scaler type: {scaler_type}, using standard")
            scaler = StandardScaler()

        df[available_cols] = scaler.fit_transform(df[available_cols])

        scalers[scaler_type] = {
            "columns": available_cols,
            "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
            "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            "data_min": scaler.data_min_.tolist()
            if hasattr(scaler, "data_min_")
            else None,
            "data_max": scaler.data_max_.tolist()
            if hasattr(scaler, "data_max_")
            else None,
        }

        print(
            f"Scaled {len(available_cols)} numerical columns using {scaler_type} scaler"
        )
        return df, scalers

    try:
        print(f"Starting preprocessing for dataset: {input_dataset}")

        # In a real implementation, load data from the ingestion component output
        # For now, simulate loading data
        print("Loading data from ingestion component...")

        # Create sample time-series data for demonstration
        date_range = pd.date_range(start="2023-01-01", end="2024-01-01", freq="H")
        sample_data = {
            timestamp_column: date_range,
            "value": np.random.randn(len(date_range)) * 100 + 1000,
            "category": np.random.choice(["A", "B", "C"], len(date_range)),
            "feature_1": np.random.randn(len(date_range)),
            "feature_2": np.random.randn(len(date_range)) * 50,
        }
        df = pd.DataFrame(sample_data)

        print(f"Loaded {len(df)} rows for preprocessing")

        # Get preprocessing configuration
        config = preprocessing_config or {}

        # Time-series feature engineering
        df = create_time_features(df, timestamp_column)

        # Create lag features if target column specified
        if target_column and target_column in df.columns:
            lags = config.get("lags", [1, 7, 24])  # Default lags for hourly data
            df = create_lag_features(df, target_column, lags)

            # Create rolling features
            windows = config.get(
                "rolling_windows", [7, 24, 168]
            )  # 7h, 24h, 7d for hourly data
            df = create_rolling_features(df, target_column, windows)

        # Handle categorical encoding
        categorical_cols = config.get("categorical_columns", ["category"])
        df, encoders = encode_categorical_features(df, categorical_cols)

        # Handle numerical scaling
        numerical_cols = config.get(
            "numerical_columns", ["feature_1", "feature_2", "value"]
        )
        scaler_type = config.get("scaler_type", "standard")
        df, scalers = scale_numerical_features(df, numerical_cols, scaler_type)

        # Remove rows with NaN values created by lag/rolling features
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        removed_rows = initial_rows - final_rows
        print(f"Removed {removed_rows} rows with NaN values after feature engineering")

        # Store processed data (simulated GCS path)
        processed_path = f"gs://{project_id}-ml-data/processed/timeseries_data.parquet"
        print(f"Processed data would be saved to: {processed_path}")

        # Prepare metadata
        feature_names = df.columns.tolist()
        metadata = {
            "rows_processed": len(df),
            "features_created": len(feature_names),
            "original_features": len(sample_data),
            "processing_timestamp": datetime.now().isoformat(),
            "timestamp_column": timestamp_column,
            "target_column": target_column,
            "categorical_encoders": encoders,
            "data_shape": df.shape,
            "preprocessing_config": config,
        }

        # Scaler information
        scaler_info = {
            "scalers": scalers,
            "scaler_type": scaler_type,
            "scaled_columns": scalers.get(scaler_type, {}).get("columns", []),
        }

        print("Time-series preprocessing completed successfully")
        print(f"Final dataset shape: {df.shape}")
        feature_preview = feature_names[:10]
        suffix = "..." if len(feature_names) > 10 else ""
        print(f"Feature names: {feature_preview}{suffix}")

        Outputs = namedtuple(
            "Outputs",
            ["processed_data_path", "metadata", "feature_names", "scaler_info"],
        )
        return Outputs(processed_path, metadata, feature_names, scaler_info)

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")

        error_metadata = {
            "rows_processed": 0,
            "features_created": 0,
            "processing_timestamp": datetime.now().isoformat(),
            "error": str(e),
        }

        Outputs = namedtuple(
            "Outputs",
            ["processed_data_path", "metadata", "feature_names", "scaler_info"],
        )
        return Outputs("", error_metadata, [], {})


@component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform", "scikit-learn", "pandas"],
)
def model_training_component(
    processed_data_path: str, model_config: dict
) -> NamedTuple("Outputs", [("model_path", str), ("metrics", dict)]):
    """Train ML model."""
    from collections import namedtuple

    print(f"Training model with data from: {processed_data_path}")
    print(f"Model config: {model_config}")

    # In a real implementation, this would:
    # - Load processed data
    # - Train the model
    # - Save model artifacts to GCS
    # - Return model metrics

    model_path = f"gs://models/trained_model_{model_config.get('version', 'v1')}"
    metrics = {"accuracy": 0.95, "precision": 0.93, "recall": 0.92, "f1_score": 0.925}

    Outputs = namedtuple("Outputs", ["model_path", "metrics"])
    return Outputs(model_path, metrics)


@component(
    base_image="python:3.11-slim", packages_to_install=["google-cloud-aiplatform"]
)
def model_evaluation_component(model_path: str, test_data_path: str) -> NamedTuple(
    "Outputs", [("evaluation_metrics", dict), ("deploy_decision", bool)]
):
    """Evaluate trained model and decide on deployment."""
    from collections import namedtuple

    print(f"Evaluating model: {model_path}")
    print(f"Test data: {test_data_path}")

    # In a real implementation, this would:
    # - Load the trained model
    # - Run evaluation on test dataset
    # - Compare against baseline metrics
    # - Make deployment decision

    evaluation_metrics = {
        "test_accuracy": 0.94,
        "test_precision": 0.92,
        "test_recall": 0.91,
        "baseline_comparison": "improved",
    }

    # Deploy if accuracy > 0.9
    deploy_decision = evaluation_metrics["test_accuracy"] > 0.9

    Outputs = namedtuple("Outputs", ["evaluation_metrics", "deploy_decision"])
    return Outputs(evaluation_metrics, deploy_decision)


@pipeline(
    name="thrasio-ml-pipeline",
    description="ML Pipeline for Thrasio data modernization project",
)
def thrasio_ml_pipeline(
    project_id: str,
    region: str,
    bq_table_name: str = "project.dataset.thrasio_raw_data",
    model_version: str = "v1",
    validation_config: dict = None,
    preprocessing_config: dict = None,
    timestamp_column: str = "timestamp",
    target_column: str = "value",
) -> NamedTuple(
    "PipelineOutputs",
    [
        ("model_path", str),
        ("final_metrics", dict),
        ("deploy_approved", bool),
        ("ingestion_results", dict),
        ("preprocessing_results", dict),
    ],
):
    """Complete ML pipeline for Thrasio project."""

    # Data ingestion step - NEW: Ingest from BigQuery with validation
    ingestion_task = data_ingestion_component(
        project_id=project_id,
        bq_table_name=bq_table_name,
        region=region,
        validation_config=validation_config
        or {"min_rows": 100, "required_columns": [], "null_thresholds": {}},
    )

    # Data preprocessing step - Enhanced with time-series features
    preprocessing_task = data_preprocessing_component(
        project_id=project_id,
        region=region,
        input_dataset=ingestion_task.outputs["ingested_data_path"],
        preprocessing_config=preprocessing_config
        or {
            "lags": [1, 7, 24, 168],  # 1h, 7h, 1d, 1w for hourly data
            "rolling_windows": [7, 24, 168],  # 7h, 1d, 1w
            "categorical_columns": ["category"],
            "numerical_columns": ["feature_1", "feature_2", "value"],
            "scaler_type": "standard",
        },
        timestamp_column=timestamp_column,
        target_column=target_column,
    )

    # Model training step
    model_config = {"version": model_version, "algorithm": "random_forest"}
    training_task = model_training_component(
        processed_data_path=preprocessing_task.outputs["processed_data_path"],
        model_config=model_config,
    )

    # Model evaluation step
    evaluation_task = model_evaluation_component(
        model_path=training_task.outputs["model_path"],
        test_data_path=preprocessing_task.outputs["processed_data_path"],
    )

    from collections import namedtuple

    PipelineOutputs = namedtuple(
        "PipelineOutputs",
        [
            "model_path",
            "final_metrics",
            "deploy_approved",
            "ingestion_results",
            "preprocessing_results",
        ],
    )
    return PipelineOutputs(
        training_task.outputs["model_path"],
        evaluation_task.outputs["evaluation_metrics"],
        evaluation_task.outputs["deploy_decision"],
        ingestion_task.outputs["validation_results"],
        preprocessing_task.outputs["metadata"],
    )


def compile_pipeline(output_path: str = "pipeline.json"):
    """Compile the pipeline definition."""
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=thrasio_ml_pipeline, package_path=output_path
    )
    print(f"Pipeline compiled to {output_path}")


def deploy_pipeline(
    project_id: str,
    region: str,
    pipeline_definition_path: str = "pipeline.json",
    pipeline_display_name: str = "thrasio-ml-pipeline",
):
    """Deploy pipeline to Vertex AI."""
    aiplatform.init(project=project_id, location=region)

    job = aiplatform.PipelineJob(
        display_name=pipeline_display_name,
        template_path=pipeline_definition_path,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "bq_table_name": f"{project_id}.thrasio_dataset.production_data",
        },
    )

    job.submit()
    print(f"Pipeline submitted: {job.resource_name}")
    return job.resource_name


if __name__ == "__main__":
    # Compile pipeline for CI/CD
    compile_pipeline()
