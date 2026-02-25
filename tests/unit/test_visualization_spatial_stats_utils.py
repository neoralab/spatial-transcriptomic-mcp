"""Unit tests for spatial statistics visualization contracts."""

from __future__ import annotations

import sys
import warnings
from types import ModuleType, SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import spatial_stats as viz_ss
from chatspatial.utils.exceptions import DataError, DataNotFoundError, ParameterError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)


def _fake_squidpy_module(
    *,
    nhood=None,
    co_occurrence=None,
    ripley=None,
    centrality=None,
) -> ModuleType:
    fake_sq = ModuleType("squidpy")
    fake_sq.pl = SimpleNamespace(
        nhood_enrichment=nhood or (lambda *a, **k: None),
        co_occurrence=co_occurrence or (lambda *a, **k: None),
        ripley=ripley or (lambda *a, **k: None),
        centrality_scores=centrality or (lambda *a, **k: None),
    )
    return fake_sq


def test_resolve_cluster_key_param_metadata_and_error(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    assert viz_ss._resolve_cluster_key(adata, "neighborhood", "group") == "group"

    adata.uns["spatial_stats_neighborhood_metadata"] = {"parameters": {"cluster_key": "group"}}
    assert viz_ss._resolve_cluster_key(adata, "neighborhood", None) == "group"

    adata2 = minimal_spatial_adata.copy()
    adata2.obs = pd.DataFrame(index=adata2.obs.index)
    with pytest.raises(ParameterError, match="cluster_key required"):
        viz_ss._resolve_cluster_key(adata2, "neighborhood", None)

    with pytest.raises(DataError, match="Cluster key not found"):
        viz_ss._resolve_cluster_key(adata, "neighborhood", "missing")


@pytest.mark.asyncio
async def test_create_spatial_statistics_visualization_routes_all_subtypes(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _fake_async(*_args, **_kwargs):
        return sentinel

    def _fake_sync(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_ss, "_create_neighborhood_enrichment_visualization", _fake_async)
    monkeypatch.setattr(viz_ss, "_create_co_occurrence_visualization", _fake_async)
    monkeypatch.setattr(viz_ss, "_create_ripley_visualization", _fake_async)
    monkeypatch.setattr(viz_ss, "_create_moran_visualization", _fake_sync)
    monkeypatch.setattr(viz_ss, "_create_centrality_visualization", _fake_async)
    monkeypatch.setattr(viz_ss, "_create_getis_ord_visualization", _fake_async)

    for subtype in [
        "neighborhood",
        "co_occurrence",
        "ripley",
        "moran",
        "centrality",
        "getis_ord",
    ]:
        out = await viz_ss.create_spatial_statistics_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="statistics", subtype=subtype),
            context=DummyCtx(),
        )
        assert out is sentinel

    with pytest.raises(ParameterError, match="Unsupported subtype for spatial_statistics"):
        await viz_ss.create_spatial_statistics_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="statistics", subtype="unknown"),
        )


@pytest.mark.asyncio
async def test_neighborhood_visualization_validates_data_and_calls_squidpy(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    monkeypatch.setattr(viz_ss, "require", lambda *_a, **_k: None)

    with pytest.raises(DataNotFoundError, match="Neighborhood enrichment not found"):
        await viz_ss._create_neighborhood_enrichment_visualization(
            adata,
            VisualizationParameters(plot_type="statistics", subtype="neighborhood", cluster_key="group"),
        )

    adata.uns["group_nhood_enrichment"] = {"dummy": True}
    called: dict[str, object] = {}

    def _nhood(*_args, **kwargs):
        called["cluster_key"] = kwargs.get("cluster_key")
        called["title"] = kwargs.get("title")

    monkeypatch.setitem(
        sys.modules,
        "squidpy",
        _fake_squidpy_module(nhood=_nhood),
    )

    fig, ax = plt.subplots()
    monkeypatch.setattr(viz_ss, "create_figure_from_params", lambda *_a, **_k: (fig, [ax]))

    out = await viz_ss._create_neighborhood_enrichment_visualization(
        adata,
        VisualizationParameters(plot_type="statistics", subtype="neighborhood", cluster_key="group"),
    )
    assert out is fig
    assert called["cluster_key"] == "group"
    assert "Neighborhood Enrichment" in called["title"]
    fig.clf()


@pytest.mark.asyncio
async def test_co_occurrence_visualization_handles_missing_and_figsize_logic(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obs["group"] = adata.obs["group"].astype("category")
    monkeypatch.setattr(viz_ss, "require", lambda *_a, **_k: None)

    with pytest.raises(DataNotFoundError, match="Co-occurrence not found"):
        await viz_ss._create_co_occurrence_visualization(
            adata,
            VisualizationParameters(plot_type="statistics", subtype="co_occurrence", cluster_key="group"),
        )

    adata.uns["group_co_occurrence"] = {"dummy": True}
    called: dict[str, object] = {}

    def _co(*_args, **kwargs):
        called["clusters"] = kwargs.get("clusters")
        called["figsize"] = kwargs.get("figsize")
        plt.figure()

    monkeypatch.setitem(sys.modules, "squidpy", _fake_squidpy_module(co_occurrence=_co))

    out = await viz_ss._create_co_occurrence_visualization(
        adata,
        VisualizationParameters(plot_type="statistics", subtype="co_occurrence", cluster_key="group"),
    )
    assert called["figsize"] is None
    assert 1 <= len(called["clusters"]) <= 4
    assert out is not None
    out.clf()


@pytest.mark.asyncio
async def test_ripley_visualization_missing_and_success(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    monkeypatch.setattr(viz_ss, "require", lambda *_a, **_k: None)
    with pytest.raises(DataNotFoundError, match="Ripley results not found"):
        await viz_ss._create_ripley_visualization(
            adata,
            VisualizationParameters(plot_type="statistics", subtype="ripley", cluster_key="group"),
        )

    adata.uns["group_ripley_L"] = {"dummy": True}
    called: dict[str, object] = {}

    def _ripley(*_args, **kwargs):
        called["cluster_key"] = kwargs.get("cluster_key")
        called["mode"] = kwargs.get("mode")

    monkeypatch.setitem(sys.modules, "squidpy", _fake_squidpy_module(ripley=_ripley))
    fig, ax = plt.subplots()
    monkeypatch.setattr(viz_ss, "create_figure_from_params", lambda *_a, **_k: (fig, [ax]))

    out = await viz_ss._create_ripley_visualization(
        adata,
        VisualizationParameters(plot_type="statistics", subtype="ripley", cluster_key="group", title="Ripley Title"),
    )
    assert out is fig
    assert called["cluster_key"] == "group"
    assert called["mode"] == "L"
    assert ax.get_title() == "Ripley Title"
    fig.clf()


def test_moran_visualization_missing_and_zero_pvalue_branch(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    with pytest.raises(DataNotFoundError, match="Moran's I results not found"):
        viz_ss._create_moran_visualization(
            adata,
            VisualizationParameters(plot_type="statistics", subtype="moran"),
        )

    idx = [f"gene_{i}" for i in range(6)]
    adata.uns["moranI"] = pd.DataFrame(
        {
            "I": [0.8, 0.5, 0.2, -0.1, 0.1, 0.0],
            "pval_norm": [0.0, 1e-5, 0.04, 0.2, 0.8, 0.5],
        },
        index=idx,
    )
    fig = viz_ss._create_moran_visualization(
        adata,
        VisualizationParameters(plot_type="statistics", subtype="moran", show_colorbar=True),
    )
    assert len(fig.axes) >= 1
    assert "Moran's I" in fig.axes[0].get_title()
    fig.clf()


@pytest.mark.asyncio
async def test_centrality_visualization_missing_and_success(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    monkeypatch.setattr(viz_ss, "require", lambda *_a, **_k: None)
    with pytest.raises(DataNotFoundError, match="Centrality scores not found"):
        await viz_ss._create_centrality_visualization(
            adata,
            VisualizationParameters(plot_type="statistics", subtype="centrality", cluster_key="group"),
        )

    adata.uns["group_centrality_scores"] = pd.DataFrame(
        {"degree": [0.2, 0.5], "closeness": [0.3, 0.4]}
    )
    called: dict[str, object] = {}

    def _centrality(*_args, **kwargs):
        called["cluster_key"] = kwargs.get("cluster_key")
        called["figsize"] = kwargs.get("figsize")
        plt.figure()

    monkeypatch.setitem(
        sys.modules,
        "squidpy",
        _fake_squidpy_module(centrality=_centrality),
    )

    fig = await viz_ss._create_centrality_visualization(
        adata,
        VisualizationParameters(
            plot_type="statistics",
            subtype="centrality",
            cluster_key="group",
            title="Centrality Title",
        ),
        context=DummyCtx(),
    )
    assert called["cluster_key"] == "group"
    assert called["figsize"] == (10, 5)
    assert fig._suptitle is not None
    fig.clf()


@pytest.mark.asyncio
async def test_getis_ord_visualization_validation_and_success(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    with pytest.raises(DataNotFoundError, match="No Getis-Ord results found"):
        await viz_ss._create_getis_ord_visualization(
            adata,
            VisualizationParameters(plot_type="statistics", subtype="getis_ord"),
            context=DummyCtx(),
        )

    adata.obs["gene_0_getis_ord_z"] = np.linspace(-2, 2, adata.n_obs)
    adata.obs["gene_0_getis_ord_p"] = np.linspace(0.001, 0.1, adata.n_obs)
    adata.obs["gene_1_getis_ord_z"] = np.linspace(2, -2, adata.n_obs)
    adata.obs["gene_1_getis_ord_p"] = np.linspace(0.001, 0.2, adata.n_obs)

    with pytest.raises(DataNotFoundError, match="None of the specified genes have Getis-Ord results"):
        await viz_ss._create_getis_ord_visualization(
            adata,
            VisualizationParameters(
                plot_type="statistics", subtype="getis_ord", feature=["missing_gene"]
            ),
            context=DummyCtx(),
        )

    monkeypatch.setattr(viz_ss, "require_spatial_coords", lambda _a: _a.obsm["spatial"])
    monkeypatch.setattr(viz_ss, "auto_spot_size", lambda *_a, **_k: 20.0)

    ctx = DummyCtx()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig = await viz_ss._create_getis_ord_visualization(
            adata,
            VisualizationParameters(
                plot_type="statistics", subtype="getis_ord", feature=["gene_0", "gene_1"]
            ),
            context=ctx,
        )
    assert any("Plotting Getis-Ord results for 2 genes" in msg for msg in ctx.infos)
    assert len(fig.axes) >= 2
    fig.clf()
