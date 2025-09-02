"""Deployment utilities for Vertex AI pipelines."""

import os
import sys
from pathlib import Path

from app.pipeline import compile_pipeline, deploy_pipeline


def main():
    """Main deployment script for CI/CD."""
    
    # Get environment variables
    project_id = os.getenv('GCP_PROJECT_ID')
    region = os.getenv('GCP_REGION', 'us-central1')
    
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
        
        # Deploy to Vertex AI
        pipeline_resource = deploy_pipeline(
            project_id=project_id,
            region=region,
            pipeline_definition_path=pipeline_path
        )
        
        print(f"✅ Pipeline deployed successfully: {pipeline_resource}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()