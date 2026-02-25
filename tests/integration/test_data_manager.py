"""Integration tests for DefaultSpatialDataManager behavior."""

import pytest

from chatspatial.spatial_mcp_adapter import DefaultSpatialDataManager
from chatspatial.utils.exceptions import DataNotFoundError


@pytest.fixture
def manager() -> DefaultSpatialDataManager:
    """Fresh manager per test to avoid shared state."""
    return DefaultSpatialDataManager()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_manager_create_update_get(manager, minimal_spatial_adata):
    data_id = await manager.create_dataset(
        minimal_spatial_adata, prefix="custom", name="demo"
    )
    ds = await manager.get_dataset(data_id)
    assert data_id.startswith("custom_")
    assert ds["name"] == "demo"
    assert ds["adata"].n_obs == minimal_spatial_adata.n_obs
    assert ds["n_cells"] == minimal_spatial_adata.n_obs
    assert ds["n_genes"] == minimal_spatial_adata.n_vars

    subset = minimal_spatial_adata[:20, :10].copy()
    await manager.update_adata(data_id, subset)
    ds2 = await manager.get_dataset(data_id)
    assert ds2["adata"].shape == (20, 10)
    assert ds2["n_cells"] == 20
    assert ds2["n_genes"] == 10

    listed = await manager.list_datasets()
    assert listed == [
        {
            "id": data_id,
            "name": "demo",
            "type": "unknown",
            "n_cells": 20,
            "n_genes": 10,
        }
    ]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_manager_auto_generated_ids_are_unique(manager, minimal_spatial_adata):
    id1 = await manager.create_dataset(minimal_spatial_adata, prefix="dup")
    id2 = await manager.create_dataset(minimal_spatial_adata, prefix="dup")

    assert id1 != id2
    assert id1.startswith("dup_")
    assert id2.startswith("dup_")

    with pytest.raises(DataNotFoundError):
        await manager.get_dataset("does_not_exist")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_manager_save_and_get_result_contract(manager, minimal_spatial_adata):
    """save_result/get_result should preserve payload and key boundaries."""
    data_id = await manager.create_dataset(minimal_spatial_adata, prefix="d")

    result_payload = {"score": 0.91, "method": "demo"}
    await manager.save_result(data_id, "annotation", result_payload)
    fetched = await manager.get_result(data_id, "annotation")

    assert fetched == result_payload
    ds = await manager.get_dataset(data_id)
    assert "results" in ds
    assert ds["results"]["annotation"] == result_payload


@pytest.mark.integration
@pytest.mark.asyncio
async def test_data_manager_get_result_missing_paths_raise(manager, minimal_spatial_adata):
    """Missing dataset and missing result should both raise DataNotFoundError."""
    data_id = await manager.create_dataset(minimal_spatial_adata, prefix="d")

    with pytest.raises(DataNotFoundError, match="No preprocessing results found"):
        await manager.get_result(data_id, "preprocessing")

    with pytest.raises(DataNotFoundError, match="Dataset absent not found"):
        await manager.get_result("absent", "preprocessing")
