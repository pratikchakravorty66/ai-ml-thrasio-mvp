#!/bin/bash

# Deployment script for weekly ML reporting automation
# This script sets up Cloud Scheduler, Cloud Functions, and necessary GCP resources

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
REGION=${GCP_REGION:-"us-central1"}
FUNCTION_NAME="weekly-ml-report-trigger"
SCHEDULER_JOB_NAME="weekly-ml-report"
BUCKET_REPORTS="${PROJECT_ID}-ml-reports"
BUCKET_PIPELINES="${PROJECT_ID}-ml-pipelines"
BUCKET_STAGING="${PROJECT_ID}-vertex-pipelines-staging"

echo "🚀 Deploying Weekly ML Reporting Automation"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Check if required environment variables are set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: GCP_PROJECT_ID environment variable not set"
    exit 1
fi

# Enable required APIs
echo "📋 Enabling required GCP APIs..."
gcloud services enable cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    aiplatform.googleapis.com \
    storage.googleapis.com \
    pubsub.googleapis.com \
    --project=$PROJECT_ID

# Create necessary GCS buckets
echo "🪣 Creating GCS buckets..."

# Reports bucket
if ! gsutil ls gs://$BUCKET_REPORTS >/dev/null 2>&1; then
    echo "Creating reports bucket: $BUCKET_REPORTS"
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_REPORTS
    gsutil lifecycle set infrastructure/bucket_lifecycle.json gs://$BUCKET_REPORTS
else
    echo "Reports bucket already exists: $BUCKET_REPORTS"
fi

# Pipelines bucket
if ! gsutil ls gs://$BUCKET_PIPELINES >/dev/null 2>&1; then
    echo "Creating pipelines bucket: $BUCKET_PIPELINES"
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_PIPELINES
else
    echo "Pipelines bucket already exists: $BUCKET_PIPELINES"
fi

# Staging bucket (if not exists)
if ! gsutil ls gs://$BUCKET_STAGING >/dev/null 2>&1; then
    echo "Creating staging bucket: $BUCKET_STAGING"
    gsutil mb -p $PROJECT_ID -l $REGION gs://$BUCKET_STAGING
else
    echo "Staging bucket already exists: $BUCKET_STAGING"
fi

# Compile and upload reporting pipeline
echo "🔧 Compiling and uploading reporting pipeline..."
cd "$(dirname "$0")/../app"

# Compile the reporting pipeline
python reporting_pipeline.py

# Upload to GCS
gsutil cp reporting_pipeline.json gs://$BUCKET_PIPELINES/

echo "✅ Pipeline uploaded to gs://$BUCKET_PIPELINES/reporting_pipeline.json"

# Create requirements.txt for Cloud Function
echo "📦 Creating requirements.txt for Cloud Function..."
cat > requirements.txt << EOF
functions-framework==3.*
google-cloud-aiplatform==1.71.1
google-cloud-bigquery
google-cloud-storage
pandas
numpy
jinja2
sendgrid==6.*
EOF

# Deploy Cloud Function
echo "☁️ Deploying Cloud Function..."
gcloud functions deploy $FUNCTION_NAME \
    --runtime python311 \
    --trigger-http \
    --entry-point weekly_report_trigger \
    --source . \
    --memory 1GB \
    --timeout 540s \
    --region $REGION \
    --project $PROJECT_ID \
    --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,GCP_REGION=$REGION,PIPELINE_TEMPLATE_PATH=gs://$BUCKET_PIPELINES/reporting_pipeline.json,REPORT_BUCKET=$BUCKET_REPORTS,STAGING_BUCKET=gs://$BUCKET_STAGING" \
    --allow-unauthenticated

# Get the Cloud Function URL
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --region=$REGION --project=$PROJECT_ID --format="value(httpsTrigger.url)")
echo "📍 Cloud Function URL: $FUNCTION_URL"

# Create Cloud Scheduler job
echo "⏰ Creating Cloud Scheduler job..."

# Check if job already exists and delete it
if gcloud scheduler jobs describe $SCHEDULER_JOB_NAME --location=$REGION --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "Deleting existing scheduler job..."
    gcloud scheduler jobs delete $SCHEDULER_JOB_NAME --location=$REGION --project=$PROJECT_ID --quiet
fi

# Create new scheduler job - every Monday at 9 AM UTC
gcloud scheduler jobs create http $SCHEDULER_JOB_NAME \
    --location=$REGION \
    --project=$PROJECT_ID \
    --schedule="0 9 * * MON" \
    --uri=$FUNCTION_URL \
    --http-method=GET \
    --time-zone="UTC" \
    --description="Weekly ML performance report generation for Thrasio"

echo "⏰ Scheduler job created: $SCHEDULER_JOB_NAME"
echo "Schedule: Every Monday at 9:00 AM UTC"

# Test the setup (optional)
echo ""
echo "🧪 Testing the setup..."
echo "You can manually trigger the report by running:"
echo "gcloud scheduler jobs run $SCHEDULER_JOB_NAME --location=$REGION --project=$PROJECT_ID"
echo ""
echo "Or test the Cloud Function directly:"
echo "curl -X GET $FUNCTION_URL"

# Create Pub/Sub topic for alternative triggering (optional)
echo "📬 Creating Pub/Sub topic for alternative triggering..."
TOPIC_NAME="weekly-ml-reports"

if ! gcloud pubsub topics describe $TOPIC_NAME --project=$PROJECT_ID >/dev/null 2>&1; then
    gcloud pubsub topics create $TOPIC_NAME --project=$PROJECT_ID
    echo "✅ Pub/Sub topic created: $TOPIC_NAME"
else
    echo "Pub/Sub topic already exists: $TOPIC_NAME"
fi

echo ""
echo "🎉 Weekly ML Reporting Automation deployed successfully!"
echo ""
echo "Summary:"
echo "- Cloud Function: $FUNCTION_NAME"
echo "- Scheduler Job: $SCHEDULER_JOB_NAME (runs every Monday at 9 AM UTC)"
echo "- Reports Bucket: gs://$BUCKET_REPORTS"
echo "- Pipeline Bucket: gs://$BUCKET_PIPELINES"
echo "- Function URL: $FUNCTION_URL"
echo ""
echo "Next Steps:"
echo "1. Set up email configuration (SENDGRID_API_KEY, EMAIL_RECIPIENTS)"
echo "2. Configure your email domain in cloud_function.py"
echo "3. Test the automation: gcloud scheduler jobs run $SCHEDULER_JOB_NAME --location=$REGION --project=$PROJECT_ID"
echo "4. Monitor execution in Cloud Functions and Vertex AI Pipelines"