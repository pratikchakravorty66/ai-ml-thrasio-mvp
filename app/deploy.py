"""Deployment utilities for Vertex AI pipelines."""

import os
import sys
from pathlib import Path

from app.pipeline import compile_pipeline


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

        # Initialize Vertex AI with authenticated session
        from google.cloud import aiplatform
        aiplatform.init(project=project_id, location=region)

        # Deploy to Vertex AI using authenticated session
        job = aiplatform.PipelineJob(
            display_name="thrasio-ml-pipeline",
            template_path=pipeline_path,
            parameter_values={
                "project_id": project_id,
                "region": region,
                "input_dataset": "thrasio_production_data"
            }
        )
        
        job.submit()
        print(f"✅ Pipeline deployed successfully: {job.resource_name}")

    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
