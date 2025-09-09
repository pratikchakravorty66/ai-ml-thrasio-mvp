"""Main entry point for Cloud Function."""

from cloud_function import weekly_report_trigger

# Export the function for Cloud Functions
__all__ = ['weekly_report_trigger'] 