"""Vertex AI Pipeline Definition for Thrasio ML MVP."""

from typing import NamedTuple

from google.cloud import aiplatform
from kfp.dsl import component, pipeline


@component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform", "pandas", "numpy"]
)
def data_preprocessing_component(
    project_id: str,
    region: str,
    input_dataset: str
) -> NamedTuple("Outputs", [("processed_data_path", str), ("metadata", dict)]):
    """Preprocess data for ML training."""
    from collections import namedtuple
    
    # Placeholder preprocessing logic
    print(f"Processing dataset: {input_dataset}")
    
    # In a real implementation, this would:
    # - Load data from BigQuery or GCS
    # - Clean and transform data
    # - Store processed data back to GCS
    
    processed_path = f"gs://{project_id}-ml-data/processed/data.parquet"
    metadata = {
        "rows_processed": 1000,
        "features_created": 50,
        "processing_timestamp": "2024-01-01T00:00:00Z"
    }
    
    Outputs = namedtuple("Outputs", ["processed_data_path", "metadata"])
    return Outputs(processed_path, metadata)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform", "scikit-learn", "pandas"]
)
def model_training_component(
    processed_data_path: str,
    model_config: dict
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
    metrics = {
        "accuracy": 0.95,
        "precision": 0.93,
        "recall": 0.92,
        "f1_score": 0.925
    }
    
    Outputs = namedtuple("Outputs", ["model_path", "metrics"])
    return Outputs(model_path, metrics)


@component(
    base_image="python:3.11-slim",
    packages_to_install=["google-cloud-aiplatform"]
)
def model_evaluation_component(
    model_path: str,
    test_data_path: str
) -> NamedTuple("Outputs", [("evaluation_metrics", dict), ("deploy_decision", bool)]):
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
        "baseline_comparison": "improved"
    }
    
    # Deploy if accuracy > 0.9
    deploy_decision = evaluation_metrics["test_accuracy"] > 0.9
    
    Outputs = namedtuple("Outputs", ["evaluation_metrics", "deploy_decision"])
    return Outputs(evaluation_metrics, deploy_decision)


@pipeline(
    name="thrasio-ml-pipeline",
    description="ML Pipeline for Thrasio data modernization project"
)
def thrasio_ml_pipeline(
    project_id: str,
    region: str,
    input_dataset: str = "thrasio_raw_data",
    model_version: str = "v1"
) -> NamedTuple(
    "PipelineOutputs", 
    [("model_path", str), ("final_metrics", dict), ("deploy_approved", bool)]
):
    """Complete ML pipeline for Thrasio project."""
    
    # Data preprocessing step
    preprocessing_task = data_preprocessing_component(
        project_id=project_id,
        region=region,
        input_dataset=input_dataset
    )
    
    # Model training step
    model_config = {"version": model_version, "algorithm": "random_forest"}
    training_task = model_training_component(
        processed_data_path=preprocessing_task.outputs["processed_data_path"],
        model_config=model_config
    )
    
    # Model evaluation step
    evaluation_task = model_evaluation_component(
        model_path=training_task.outputs["model_path"],
        test_data_path=preprocessing_task.outputs["processed_data_path"]
    )
    
    from collections import namedtuple
    
    PipelineOutputs = namedtuple(
        "PipelineOutputs", ["model_path", "final_metrics", "deploy_approved"]
    )
    return PipelineOutputs(
        training_task.outputs["model_path"],
        evaluation_task.outputs["evaluation_metrics"],
        evaluation_task.outputs["deploy_decision"]
    )


def compile_pipeline(output_path: str = "pipeline.json"):
    """Compile the pipeline definition."""
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=thrasio_ml_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled to {output_path}")


def deploy_pipeline(
    project_id: str,
    region: str,
    pipeline_definition_path: str = "pipeline.json",
    pipeline_display_name: str = "thrasio-ml-pipeline"
):
    """Deploy pipeline to Vertex AI."""
    aiplatform.init(project=project_id, location=region)
    
    job = aiplatform.PipelineJob(
        display_name=pipeline_display_name,
        template_path=pipeline_definition_path,
        parameter_values={
            "project_id": project_id,
            "region": region,
            "input_dataset": "thrasio_production_data"
        }
    )
    
    job.submit()
    print(f"Pipeline submitted: {job.resource_name}")
    return job.resource_name


if __name__ == "__main__":
    # Compile pipeline for CI/CD
    compile_pipeline()