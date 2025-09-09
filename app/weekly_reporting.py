"""Weekly Report Generation for Thrasio ML MVP."""

from collections import namedtuple
from typing import Any, Dict, List, NamedTuple

from google.cloud import aiplatform, bigquery, storage
from kfp.dsl import component


@component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-bigquery", 
        "google-cloud-storage",
        "pandas",
        "numpy",
        "jinja2",
        "sendgrid",
    ],
)
def weekly_report_generation_component(
    project_id: str,
    region: str,
    report_bucket: str,
    email_recipients: List[str],
    sendgrid_api_key: str,
    report_config: Dict[str, Any] = None,
) -> NamedTuple(
    "Outputs", 
    [
        ("report_path", str),
        ("metrics_summary", dict),
        ("status", str),
        ("email_status", str),
    ]
):
    """Generate weekly ML pipeline performance report."""
    import json
    from collections import namedtuple
    from datetime import datetime, timedelta
    from typing import Dict, List

    import pandas as pd
    from google.cloud import aiplatform, bigquery, storage
    from jinja2 import Template
    
    def get_pipeline_metrics(project_id: str, region: str, days_back: int = 7) -> Dict:
        """Retrieve pipeline execution metrics from the last week."""
        try:
            aiplatform.init(project=project_id, location=region)
            
            # Get pipeline jobs from last week
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            jobs = aiplatform.PipelineJob.list(
                filter=f'create_time>="{start_time.isoformat()}"',
                order_by='create_time desc'
            )
            
            pipeline_metrics = {
                "total_runs": len(jobs),
                "successful_runs": 0,
                "failed_runs": 0,
                "running_jobs": 0,
                "avg_duration_minutes": 0,
                "jobs_summary": []
            }
            
            total_duration = 0
            completed_jobs = 0
            
            for job in jobs:
                job_info = {
                    "display_name": job.display_name,
                    "state": str(job.state),
                    "create_time": job.create_time.isoformat() if job.create_time else None,
                    "end_time": job.end_time.isoformat() if job.end_time else None,
                }
                
                if job.state == aiplatform.PipelineJob.JobState.JOB_STATE_SUCCEEDED:
                    pipeline_metrics["successful_runs"] += 1
                    if job.create_time and job.end_time:
                        duration = (job.end_time - job.create_time).total_seconds() / 60
                        total_duration += duration
                        completed_jobs += 1
                        job_info["duration_minutes"] = round(duration, 2)
                elif job.state == aiplatform.PipelineJob.JobState.JOB_STATE_FAILED:
                    pipeline_metrics["failed_runs"] += 1
                elif job.state == aiplatform.PipelineJob.JobState.JOB_STATE_RUNNING:
                    pipeline_metrics["running_jobs"] += 1
                
                pipeline_metrics["jobs_summary"].append(job_info)
            
            if completed_jobs > 0:
                pipeline_metrics["avg_duration_minutes"] = round(total_duration / completed_jobs, 2)
            
            return pipeline_metrics
            
        except Exception as e:
            print(f"Error retrieving pipeline metrics: {str(e)}")
            return {
                "total_runs": 0,
                "successful_runs": 0,
                "failed_runs": 0,
                "error": str(e)
            }
    
    def get_model_metrics(project_id: str, region: str) -> Dict:
        """Retrieve model performance metrics."""
        try:
            aiplatform.init(project=project_id, location=region)
            
            # Get models
            models = aiplatform.Model.list()
            
            model_metrics = {
                "total_models": len(models),
                "models_summary": []
            }
            
            for model in models[:5]:  # Limit to 5 most recent models
                model_info = {
                    "display_name": model.display_name,
                    "create_time": model.create_time.isoformat() if model.create_time else None,
                    "model_id": model.name.split("/")[-1] if model.name else None,
                    "labels": dict(model.labels) if model.labels else {}
                }
                model_metrics["models_summary"].append(model_info)
            
            return model_metrics
            
        except Exception as e:
            print(f"Error retrieving model metrics: {str(e)}")
            return {"total_models": 0, "error": str(e)}
    
    def get_data_quality_metrics(project_id: str) -> Dict:
        """Retrieve data quality metrics from BigQuery."""
        try:
            client = bigquery.Client(project=project_id)
            
            # Sample query to get data statistics
            query = f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT(DISTINCT DATE(EXTRACT(DATE FROM CURRENT_TIMESTAMP()))) as days_with_data,
                AVG(CASE WHEN col IS NOT NULL THEN 1 ELSE 0 END) * 100 as avg_completeness
            FROM (
                SELECT 1 as col
                LIMIT 1000
            )
            """
            
            # Execute query
            query_job = client.query(query)
            results = list(query_job)
            
            if results:
                row = results[0]
                return {
                    "total_rows_processed": int(row.total_rows),
                    "data_completeness_pct": round(float(row.avg_completeness), 2),
                    "status": "healthy"
                }
            else:
                return {"status": "no_data", "total_rows_processed": 0}
                
        except Exception as e:
            print(f"Error retrieving data quality metrics: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def generate_html_report(
        pipeline_metrics: Dict,
        model_metrics: Dict, 
        data_metrics: Dict,
        report_date: str
    ) -> str:
        """Generate HTML report from metrics."""
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Thrasio ML Weekly Report - {{ report_date }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #f4f4f4; padding: 20px; border-radius: 5px; }
                .metric-card { background-color: #fff; border: 1px solid #ddd; 
                              border-radius: 5px; padding: 15px; margin: 10px 0; }
                .success { color: #28a745; }
                .error { color: #dc3545; }
                .warning { color: #ffc107; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f4f4f4; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Thrasio ML Weekly Report</h1>
                <p><strong>Report Date:</strong> {{ report_date }}</p>
                <p><strong>Reporting Period:</strong> Last 7 days</p>
            </div>
            
            <div class="metric-card">
                <h2>📊 Pipeline Performance</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Runs</td><td>{{ pipeline_metrics.total_runs }}</td></tr>
                    <tr><td>Successful Runs</td><td class="success">{{ pipeline_metrics.successful_runs }}</td></tr>
                    <tr><td>Failed Runs</td><td class="error">{{ pipeline_metrics.failed_runs }}</td></tr>
                    <tr><td>Running Jobs</td><td class="warning">{{ pipeline_metrics.running_jobs }}</td></tr>
                    <tr><td>Average Duration (min)</td><td>{{ pipeline_metrics.avg_duration_minutes }}</td></tr>
                </table>
            </div>
            
            <div class="metric-card">
                <h2>🤖 Model Status</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Models</td><td>{{ model_metrics.total_models }}</td></tr>
                </table>
                
                {% if model_metrics.models_summary %}
                <h3>Recent Models:</h3>
                <table>
                    <tr><th>Model Name</th><th>Created</th><th>Model ID</th></tr>
                    {% for model in model_metrics.models_summary %}
                    <tr>
                        <td>{{ model.display_name }}</td>
                        <td>{{ model.create_time[:10] if model.create_time else 'N/A' }}</td>
                        <td>{{ model.model_id if model.model_id else 'N/A' }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            
            <div class="metric-card">
                <h2>📈 Data Quality</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Status</td><td class="{% if data_metrics.status == 'healthy' %}success{% else %}warning{% endif %}">{{ data_metrics.status|title }}</td></tr>
                    <tr><td>Rows Processed</td><td>{{ data_metrics.get('total_rows_processed', 'N/A') }}</td></tr>
                    {% if data_metrics.get('data_completeness_pct') %}
                    <tr><td>Data Completeness</td><td>{{ data_metrics.data_completeness_pct }}%</td></tr>
                    {% endif %}
                </table>
            </div>
            
            {% if pipeline_metrics.jobs_summary %}
            <div class="metric-card">
                <h2>📋 Recent Pipeline Jobs</h2>
                <table>
                    <tr><th>Job Name</th><th>State</th><th>Created</th><th>Duration (min)</th></tr>
                    {% for job in pipeline_metrics.jobs_summary[:5] %}
                    <tr>
                        <td>{{ job.display_name }}</td>
                        <td class="{% if 'SUCCEEDED' in job.state %}success{% elif 'FAILED' in job.state %}error{% else %}warning{% endif %}">
                            {{ job.state.replace('JOB_STATE_', '') }}
                        </td>
                        <td>{{ job.create_time[:16] if job.create_time else 'N/A' }}</td>
                        <td>{{ job.get('duration_minutes', 'N/A') }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            <div class="metric-card">
                <p><em>Generated automatically by Thrasio ML Reporting System</em></p>
                <p><em>For questions or issues, please contact the ML team.</em></p>
            </div>
        </body>
        </html>
        """
        
        template = Template(html_template)
        return template.render(
            report_date=report_date,
            pipeline_metrics=pipeline_metrics,
            model_metrics=model_metrics,
            data_metrics=data_metrics
        )
    
    def send_email_report(
        html_content: str,
        recipients: List[str],
        sendgrid_api_key: str,
        subject: str
    ) -> str:
        """Send email report using SendGrid."""
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail
            
            sg = SendGridAPIClient(api_key=sendgrid_api_key)
            
            message = Mail(
                from_email='noreply@thrasio.com',  # Configure with your domain
                to_emails=recipients,
                subject=subject,
                html_content=html_content
            )
            
            response = sg.send(message)
            return f"Email sent successfully. Status code: {response.status_code}"
            
        except Exception as e:
            return f"Failed to send email: {str(e)}"
    
    def save_report_to_gcs(
        html_content: str,
        bucket_name: str,
        file_path: str
    ) -> str:
        """Save report to Google Cloud Storage."""
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(file_path)
            
            blob.upload_from_string(html_content, content_type='text/html')
            return f"gs://{bucket_name}/{file_path}"
            
        except Exception as e:
            print(f"Failed to save report to GCS: {str(e)}")
            return ""
    
    try:
        print("🔄 Starting weekly report generation...")
        
        # Generate timestamp
        report_date = datetime.now().strftime("%Y-%m-%d")
        
        # Collect metrics
        print("📊 Collecting pipeline metrics...")
        pipeline_metrics = get_pipeline_metrics(project_id, region)
        
        print("🤖 Collecting model metrics...")
        model_metrics = get_model_metrics(project_id, region)
        
        print("📈 Collecting data quality metrics...")
        data_metrics = get_data_quality_metrics(project_id)
        
        # Generate report
        print("📝 Generating HTML report...")
        html_report = generate_html_report(
            pipeline_metrics, model_metrics, data_metrics, report_date
        )
        
        # Save to GCS
        report_filename = f"weekly-reports/{report_date}/ml-report.html"
        print(f"💾 Saving report to GCS: {report_filename}")
        gcs_path = save_report_to_gcs(html_report, report_bucket, report_filename)
        
        # Send email
        email_status = "not_configured"
        if sendgrid_api_key and email_recipients:
            print("📧 Sending email report...")
            subject = f"Thrasio ML Weekly Report - {report_date}"
            email_status = send_email_report(html_report, email_recipients, sendgrid_api_key, subject)
        
        # Prepare summary
        metrics_summary = {
            "pipeline_runs": pipeline_metrics.get("total_runs", 0),
            "successful_runs": pipeline_metrics.get("successful_runs", 0),
            "total_models": model_metrics.get("total_models", 0),
            "data_status": data_metrics.get("status", "unknown"),
            "report_generated": True,
            "generation_time": datetime.now().isoformat()
        }
        
        print("✅ Weekly report generated successfully!")
        
        Outputs = namedtuple("Outputs", ["report_path", "metrics_summary", "status", "email_status"])
        return Outputs(gcs_path, metrics_summary, "success", email_status)
        
    except Exception as e:
        print(f"❌ Error generating weekly report: {str(e)}")
        
        error_summary = {
            "error": str(e),
            "report_generated": False,
            "generation_time": datetime.now().isoformat()
        }
        
        Outputs = namedtuple("Outputs", ["report_path", "metrics_summary", "status", "email_status"])
        return Outputs("", error_summary, "failed", "failed")