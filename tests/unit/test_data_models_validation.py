"""Unit tests for Pydantic parameter model validation contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from chatspatial.models.data import (
    AnnotationParameters,
    DifferentialExpressionParameters,
    PreprocessingParameters,
    VisualizationParameters,
)


def test_differential_expression_requires_group_key():
    with pytest.raises(ValidationError):
        DifferentialExpressionParameters()


def test_differential_expression_rejects_invalid_method():
    with pytest.raises(ValidationError):
        DifferentialExpressionParameters(group_key="cluster", method="unknown")


def test_preprocessing_rejects_invalid_normalization():
    with pytest.raises(ValidationError):
        PreprocessingParameters(normalization="invalid")


def test_preprocessing_rejects_out_of_range_scrublet_rate():
    with pytest.raises(ValidationError):
        PreprocessingParameters(scrublet_expected_doublet_rate=0.9)


def test_visualization_preprocesses_string_input():
    params = VisualizationParameters.model_validate("gene:CCL21")
    assert params.plot_type == "feature"
    assert params.feature == "CCL21"


def test_visualization_preprocess_params_handles_none_alias_and_passthrough():
    assert VisualizationParameters.preprocess_params(None) == {}
    assert VisualizationParameters.preprocess_params("CXCL12") == {
        "feature": "CXCL12",
        "plot_type": "feature",
    }
    assert VisualizationParameters.preprocess_params({"features": ["g1", "g2"]}) == {
        "feature": ["g1", "g2"]
    }
    sentinel = object()
    assert VisualizationParameters.preprocess_params(sentinel) is sentinel


def test_visualization_statistics_requires_subtype():
    with pytest.raises(ValidationError, match="subtype is required when plot_type='statistics'"):
        VisualizationParameters(plot_type="statistics")


@pytest.mark.parametrize(
    ("plot_type", "expected_subtype"),
    [
        ("cnv", "heatmap"),
        ("velocity", "stream"),
        ("enrichment", "barplot"),
        ("trajectory", "pseudotime"),
    ],
)
def test_visualization_defaults_subtype_by_plot_type(plot_type: str, expected_subtype: str):
    params = VisualizationParameters(plot_type=plot_type)
    assert params.subtype == expected_subtype


def test_visualization_communication_defaults_to_dotplot():
    params = VisualizationParameters(plot_type="communication")
    assert params.subtype == "dotplot"


def test_annotation_rejects_invalid_method_literal():
    with pytest.raises(ValidationError):
        AnnotationParameters(method="not_supported")
