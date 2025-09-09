# Weekly ML Reporting Automation

Automated system for generating and delivering weekly ML performance reports for the Thrasio AI/ML project.

## Overview

The system consists of:

1. **Report Generation Pipeline** (`app/weekly_reporting.py`) - KFP component that collects metrics and generates HTML reports
2. **Reporting Pipeline** (`app/reporting_pipeline.py`) - Vertex AI pipeline orchestrating the report generation
3. **Cloud Function** (`app/cloud_function.py`) - HTTP endpoint triggered by Cloud Scheduler
4. **Infrastructure** (`infrastructure/`) - Deployment scripts and configuration
5. **Automation** - Cloud Scheduler running weekly (Mondays at 9 AM UTC)

## Architecture

```
Cloud Scheduler (Weekly) 
    ↓
Cloud Function (HTTP Trigger)
    ↓
Vertex AI Pipeline Job
    ↓
Report Generation Component
    ↓
HTML Report → GCS Storage + Email Delivery
```

## Features

### Report Contents
- **Pipeline Performance**: Success/failure rates, execution times, recent jobs
- **Model Metrics**: Active models, deployment status
- **Data Quality**: Row counts, completeness metrics, health status
- **Visual Dashboard**: HTML report with tables and status indicators

### Delivery Options
- **Cloud Storage**: Reports saved to GCS with lifecycle management
- **Email**: HTML reports sent via SendGrid to configured recipients
- **Scheduling**: Automated weekly execution every Monday at 9 AM UTC

## Setup and Deployment

### 1. Prerequisites
```bash
# Install dependencies
poetry install --with dev

# Set environment variables
export GCP_PROJECT_ID="your-project-id"
export GCP_REGION="us-central1"
export EMAIL_RECIPIENTS="team@thrasio.com,ml-team@thrasio.com"
export SENDGRID_API_KEY="your-sendgrid-key"
```

### 2. Deploy Automation
```bash
# Make deployment script executable
chmod +x infrastructure/deploy_reporting.sh

# Deploy all components
./infrastructure/deploy_reporting.sh
```

### 3. Manual Testing
```bash
# Test pipeline compilation
python tests/test_reporting.py

# Manually trigger Cloud Scheduler job
gcloud scheduler jobs run weekly-ml-report --location=us-central1 --project=$GCP_PROJECT_ID

# Test Cloud Function directly
curl -X GET https://us-central1-$GCP_PROJECT_ID.cloudfunctions.net/weekly-ml-report-trigger
```

## Configuration

### Environment Variables (Cloud Function)
- `GCP_PROJECT_ID`: Your GCP project ID
- `GCP_REGION`: GCP region (default: us-central1)
- `PIPELINE_TEMPLATE_PATH`: GCS path to compiled pipeline
- `REPORT_BUCKET`: GCS bucket for storing reports
- `STAGING_BUCKET`: Vertex AI pipeline staging bucket
- `EMAIL_RECIPIENTS`: Comma-separated list of email addresses
- `SENDGRID_API_KEY`: SendGrid API key for email delivery

### Report Configuration
```python
report_config = {
    "include_pipeline_metrics": True,
    "include_model_metrics": True,
    "include_data_quality": True,
    "days_back": 7,  # Look back period for metrics
}
```

## Monitoring and Maintenance

### Logs and Monitoring
- **Cloud Functions**: Check execution logs in Cloud Console
- **Vertex AI**: Monitor pipeline execution in Vertex AI Pipelines
- **Cloud Scheduler**: View job history and execution status
- **Email Delivery**: Check SendGrid delivery logs

### Storage Management
- Reports are automatically moved to cheaper storage classes after 30/90 days
- Reports are deleted after 365 days (configurable in `bucket_lifecycle.json`)

### Troubleshooting

**Pipeline Compilation Errors:**
```bash
cd app
python reporting_pipeline.py
# Check for import or syntax errors
```

**Cloud Function Deployment Issues:**
```bash
# Check function logs
gcloud functions logs read weekly-ml-report-trigger --region=us-central1

# Redeploy function
./infrastructure/deploy_reporting.sh
```

**Email Delivery Problems:**
- Verify `SENDGRID_API_KEY` is valid
- Check email addresses in `EMAIL_RECIPIENTS`
- Review SendGrid domain authentication

## Customization

### Adding New Metrics
Edit `app/weekly_reporting.py`:
```python
def get_custom_metrics(project_id: str) -> Dict:
    # Add your custom metric collection logic
    return {"custom_metric": "value"}
```

### Modifying Report Template
Update the HTML template in `generate_html_report()` function to customize report appearance and content.

### Changing Schedule
Modify the cron expression in `infrastructure/deploy_reporting.sh`:
```bash
# Current: Every Monday at 9 AM UTC
--schedule="0 9 * * MON"

# Daily at 8 AM UTC
--schedule="0 8 * * *"

# Bi-weekly (every other Monday)
--schedule="0 9 * * MON/2"
```

## Testing

Run the test suite:
```bash
python tests/test_reporting.py
```

Tests include:
- Package requirement validation
- Pipeline compilation verification
- Cloud Function import testing

## Security Considerations

- Cloud Function uses service account authentication
- Sensitive credentials stored as environment variables
- GCS buckets have lifecycle policies for cost optimization
- Email delivery through authenticated SendGrid API

## Cost Optimization

- **Function Runtime**: 1GB memory, 9-minute timeout
- **Storage**: Automatic lifecycle transitions (Standard → Nearline → Coldline → Delete)
- **Scheduling**: Weekly execution minimizes compute costs
- **Pipeline**: Optimized components with minimal resource requirements

## Support

For issues or questions:
1. Check logs in Cloud Console
2. Review pipeline execution in Vertex AI
3. Run local tests: `python tests/test_reporting.py`
4. Contact ML team for configuration assistance