"""Cloud Function for triggering weekly ML reports."""

import json
import logging
import os
from typing import Any, Dict

import functions_framework
from google.cloud import aiplatform


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@functions_framework.http
def weekly_report_trigger(request):
    """HTTP Cloud Function to trigger weekly ML reporting pipeline.
    
    This function is designed to be triggered by Cloud Scheduler weekly.
    It initiates the Vertex AI reporting pipeline and returns status.
    """
    try:
        # Get environment variables
        project_id = os.getenv("GCP_PROJECT_ID")
        region = os.getenv("GCP_REGION", "us-central1")
        pipeline_template_path = os.getenv("PIPELINE_TEMPLATE_PATH", "gs://your-bucket/reporting_pipeline.json")
        report_bucket = os.getenv("REPORT_BUCKET", f"{project_id}-ml-reports")
        sendgrid_api_key = os.getenv("SENDGRID_API_KEY", "")
        
        # Get email recipients from environment variable (comma-separated)
        email_recipients_str = os.getenv("EMAIL_RECIPIENTS", "")
        email_recipients = [email.strip() for email in email_recipients_str.split(",")] if email_recipients_str else []
        
        if not project_id:
            return {"error": "GCP_PROJECT_ID environment variable not set"}, 500
        
        logger.info(f"Triggering weekly report for project: {project_id}")
        
        # Initialize Vertex AI
        staging_bucket = os.getenv("STAGING_BUCKET", f"gs://{project_id}-vertex-pipelines-staging")
        aiplatform.init(
            project=project_id,
            location=region,
            staging_bucket=staging_bucket
        )
        
        # Create pipeline job
        pipeline_display_name = f"weekly-ml-report-{aiplatform.utils.timestamped_unique_name()}"
        
        job = aiplatform.PipelineJob(
            display_name=pipeline_display_name,
            template_path=pipeline_template_path,
            parameter_values={
                "project_id": project_id,
                "region": region,
                "report_bucket": report_bucket,
                "email_recipients": email_recipients,
                "sendgrid_api_key": sendgrid_api_key,
                "report_config": {
                    "include_pipeline_metrics": True,
                    "include_model_metrics": True,
                    "include_data_quality": True,
                    "days_back": 7,
                }
            },
            pipeline_root=staging_bucket,
        )
        
        # Submit the pipeline job
        job.submit()
        
        logger.info(f"Pipeline job submitted successfully: {job.resource_name}")
        
        response = {
            "status": "success",
            "message": "Weekly reporting pipeline triggered successfully",
            "pipeline_job": job.resource_name,
            "pipeline_name": pipeline_display_name,
            "project_id": project_id,
            "region": region,
        }
        
        return response, 200
        
    except Exception as e:
        logger.error(f"Error triggering weekly report pipeline: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to trigger weekly report: {str(e)}"
        }, 500


@functions_framework.cloud_event
def weekly_report_trigger_pubsub(cloud_event):
    """Pub/Sub triggered Cloud Function for weekly ML reporting.
    
    Alternative trigger method using Pub/Sub instead of HTTP.
    """
    try:
        # Decode the Pub/Sub message
        import base64
        
        if 'data' in cloud_event.data:
            message_data = base64.b64decode(cloud_event.data['data']).decode('utf-8')
            logger.info(f"Received Pub/Sub message: {message_data}")
        
        # Trigger the same logic as HTTP function
        # You can reuse the logic from weekly_report_trigger function
        result = weekly_report_trigger(None)  # Pass None since we don't need request object
        
        logger.info(f"Pub/Sub triggered report completed with status: {result[1]}")
        
    except Exception as e:
        logger.error(f"Error in Pub/Sub triggered report: {str(e)}")
        raise


def main():
    """Local testing function."""
    import sys
    
    # Set environment variables for testing
    os.environ["GCP_PROJECT_ID"] = "your-project-id"  # Replace with actual project
    os.environ["GCP_REGION"] = "us-central1"
    os.environ["EMAIL_RECIPIENTS"] = "team@thrasio.com"
    
    # Test the function
    try:
        result, status_code = weekly_report_trigger(None)
        print(f"Result: {result}")
        print(f"Status Code: {status_code}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()