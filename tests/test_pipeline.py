"""Tests for Vertex AI pipeline components."""

from unittest.mock import MagicMock, patch

from app.pipeline import compile_pipeline, deploy_pipeline


def test_compile_pipeline():
    """Test pipeline compilation."""
    with patch("kfp.compiler.Compiler") as mock_compiler:
        mock_instance = MagicMock()
        mock_compiler.return_value = mock_instance

        compile_pipeline("test_pipeline.json")

        mock_compiler.assert_called_once()
        mock_instance.compile.assert_called_once()


def test_deploy_pipeline():
    """Test pipeline deployment."""
    with patch("google.cloud.aiplatform.init") as mock_init, patch(
        "google.cloud.aiplatform.PipelineJob"
    ) as mock_job_class:
        mock_job = MagicMock()
        mock_job.resource_name = "test-pipeline-resource"
        mock_job_class.return_value = mock_job

        result = deploy_pipeline(
            project_id="test-project",
            region="us-central1",
            pipeline_definition_path="test.json",
        )

        mock_init.assert_called_once_with(
            project="test-project", location="us-central1"
        )
        mock_job.submit.assert_called_once()
        assert result == "test-pipeline-resource"


def test_deploy_pipeline_integration():
    """Test that pipeline components are properly defined."""
    from app.pipeline import thrasio_ml_pipeline

    # Test that pipeline function is callable
    assert callable(thrasio_ml_pipeline)

    # Test pipeline with mock parameters
    # This would be expanded with actual integration tests
