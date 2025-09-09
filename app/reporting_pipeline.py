"""Reporting Pipeline for Thrasio ML MVP."""

from typing import NamedTuple

from kfp.dsl import pipeline


@pipeline(
    name="thrasio-weekly-reporting-pipeline",
    description="Weekly ML reporting pipeline for Thrasio data modernization project",
)
def weekly_reporting_pipeline(
    project_id: str,
    region: str = "us-central1",
    report_bucket: str = None,
    email_recipients: list = None,
    sendgrid_api_key: str = "",
    report_config: dict = None,
) -> NamedTuple(
    "ReportingOutputs",
    [
        ("report_path", str),
        ("metrics_summary", dict),
        ("status", str),
        ("email_status", str),
    ],
):
    """Weekly reporting pipeline to generate and deliver ML performance reports."""
    from collections import namedtuple

    # Import the reporting component
    from weekly_reporting import weekly_report_generation_component
    
    # Set default values
    if not report_bucket:
        report_bucket = f"{project_id}-ml-reports"
    
    if not email_recipients:
        email_recipients = []
    
    if not report_config:
        report_config = {
            "include_pipeline_metrics": True,
            "include_model_metrics": True,
            "include_data_quality": True,
            "days_back": 7,
        }
    
    # Execute reporting component
    reporting_task = weekly_report_generation_component(
        project_id=project_id,
        region=region,
        report_bucket=report_bucket,
        email_recipients=email_recipients,
        sendgrid_api_key=sendgrid_api_key,
        report_config=report_config,
    )
    
    # Return outputs
    ReportingOutputs = namedtuple(
        "ReportingOutputs",
        ["report_path", "metrics_summary", "status", "email_status"],
    )
    
    return ReportingOutputs(
        reporting_task.outputs["report_path"],
        reporting_task.outputs["metrics_summary"],
        reporting_task.outputs["status"],
        reporting_task.outputs["email_status"],
    )


def compile_reporting_pipeline(output_path: str = "reporting_pipeline.json"):
    """Compile the reporting pipeline definition."""
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=weekly_reporting_pipeline, package_path=output_path
    )
    print(f"Reporting pipeline compiled to {output_path}")


if __name__ == "__main__":
    # Compile pipeline for CI/CD
    compile_reporting_pipeline()