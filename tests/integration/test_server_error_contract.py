"""Integration tests for server-layer error contract stability."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from chatspatial.models.data import DifferentialExpressionParameters, VisualizationParameters
from chatspatial.server import (
    analyze_enrichment,
    find_markers,
    load_data,
    preprocess_data,
    visualize_data,
)
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError, ProcessingError


@pytest.mark.integration
@pytest.mark.asyncio
async def test_preprocess_data_missing_dataset_raises_data_not_found(reset_data_manager):
    with pytest.raises(DataNotFoundError, match="Dataset missing_data not found"):
        await preprocess_data("missing_data")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_find_markers_invalid_method_raises_validation_error(reset_data_manager):
    with pytest.raises(ValidationError, match="method"):
        params = DifferentialExpressionParameters(
            group_key="group", method="not_a_method",
        )
        await find_markers(data_id="any", params=params)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visualize_data_missing_dataset_raises_not_found(reset_data_manager):
    with pytest.raises(DataNotFoundError, match="Dataset missing_data not found"):
        await visualize_data(
            "missing_data",
            params=VisualizationParameters(plot_type="feature", feature="gene_0"),
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_analyze_enrichment_requires_params(reset_data_manager, monkeypatch: pytest.MonkeyPatch):
    # Keep this test focused on server-level parameter contract.
    # analyze_enrichment imports tools module before validating params,
    # so we inject a lightweight stub to avoid dependency noise.
    fake_module = SimpleNamespace(analyze_enrichment=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "chatspatial.tools.enrichment", fake_module)

    with pytest.raises(ParameterError, match="EnrichmentParameters is required"):
        await analyze_enrichment(data_id="any", params=None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_load_data_invalid_path_raises_file_not_found(reset_data_manager):
    with pytest.raises(FileNotFoundError, match="Data path not found"):
        await load_data("/definitely/not/exist/file.h5ad", "generic", name="bad")
