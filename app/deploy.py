"""Deployment utilities for Vertex AI pipelines."""

import os
import sys
from pathlib import Path

from google.cloud import aiplatform


def compile_pipeline(output_path: str = "pipeline.json"):
    """Compile the pipeline definition."""
    # Add current directory to Python path for imports
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))

    try:
        from kfp import compiler

        from app.pipeline import thrasio_ml_pipeline

        compiler.Compiler().compile(
            pipeline_func=thrasio_ml_pipeline, package_path=output_path
        )
        print(f"Pipeline compiled to {output_path}")
    except ImportError as e:
        print(f"Import error: {e}")
        raise


def main():
    """Main deployment script for CI/CD."""

    # Get environment variables
    project_id = os.getenv("GCP_PROJECT_ID")
    region = os.getenv("GCP_REGION", "us-central1")

    if not project_id:
        print("Error: GCP_PROJECT_ID environment variable not set")
        sys.exit(1)

    print(f"Deploying pipeline to project: {project_id}, region: {region}")

    try:
        # Compile pipeline
        pipeline_path = "pipeline.json"
        compile_pipeline(pipeline_path)

        # Verify pipeline file exists
        if not Path(pipeline_path).exists():
            print(f"Error: Pipeline file {pipeline_path} not found")
            sys.exit(1)

        # Initialize Vertex AI with authenticated session and staging bucket
        staging_bucket = f"gs://{project_id}-vertex-pipelines-pratik-20250903"
        aiplatform.init(
            project=project_id, location=region, staging_bucket=staging_bucket
        )

        # Deploy to Vertex AI using authenticated session
        job = aiplatform.PipelineJob(
            display_name="thrasio-ml-pipeline",
            template_path=pipeline_path,
            parameter_values={
                "project_id": project_id,
                "region": region,
                "input_dataset": "thrasio_production_data",
            },
        )

        job.submit()
        print(f"✅ Pipeline deployed successfully: {job.resource_name}")

    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
