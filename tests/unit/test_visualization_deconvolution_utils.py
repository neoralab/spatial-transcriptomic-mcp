"""Unit tests for deconvolution visualization retrieval and routing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chatspatial.models.data import VisualizationParameters
from chatspatial.tools.visualization import deconvolution as viz_deconv
from chatspatial.utils.exceptions import DataNotFoundError, ParameterError


class DummyCtx:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    async def info(self, msg: str):
        self.infos.append(msg)

    async def warning(self, msg: str):
        self.warnings.append(msg)


def _add_deconv_metadata(
    adata,
    method: str,
    *,
    proportions_key: str,
    cell_types: list[str] | None = None,
    dominant_type_key: str | None = None,
) -> None:
    adata.uns[f"deconvolution_{method}_metadata"] = {
        "parameters": {
            "proportions_key": proportions_key,
            "cell_types": cell_types,
            "dominant_type_key": dominant_type_key,
        }
    }


def _mock_deconv_data(adata, method: str = "mock"):
    proportions = pd.DataFrame(
        {
            "T": np.array([0.9, 0.2, 0.0, 0.4, 0.8][: adata.n_obs], dtype=float),
            "B": np.array([0.1, 0.8, 0.5, 0.3, 0.2][: adata.n_obs], dtype=float),
            "Myeloid": np.array([0.0, 0.0, 0.5, 0.3, 0.0][: adata.n_obs], dtype=float),
        },
        index=adata.obs_names[: min(5, adata.n_obs)],
    )
    if len(proportions) < adata.n_obs:
        # Repeat rows for larger fixtures while preserving proportions shape
        reps = int(np.ceil(adata.n_obs / len(proportions)))
        proportions = pd.concat([proportions] * reps, ignore_index=True).iloc[: adata.n_obs]
        proportions.index = adata.obs_names
    else:
        proportions = proportions.iloc[: adata.n_obs]
        proportions.index = adata.obs_names

    return viz_deconv.DeconvolutionData(
        proportions=proportions,
        method=method,
        cell_types=list(proportions.columns),
        proportions_key="deconvolution_mock",
        dominant_type_key=None,
    )


def test_get_available_methods_prefers_metadata_then_fallback(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.uns["deconvolution_a_metadata"] = {"parameters": {}}
    adata.uns["deconvolution_b_metadata"] = {"parameters": {}}
    adata.obsm["deconvolution_legacy"] = np.zeros((adata.n_obs, 2))

    methods = viz_deconv._get_available_methods(adata)
    assert set(methods) == {"a", "b"}

    adata2 = minimal_spatial_adata.copy()
    adata2.obsm["deconvolution_rctd"] = np.zeros((adata2.n_obs, 2))
    assert viz_deconv._get_available_methods(adata2) == ["rctd"]


@pytest.mark.asyncio
async def test_get_deconvolution_data_requires_existing_results(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="No deconvolution results found"):
        await viz_deconv.get_deconvolution_data(minimal_spatial_adata)


@pytest.mark.asyncio
async def test_get_deconvolution_data_requires_method_when_multiple(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.uns["deconvolution_m1_metadata"] = {"parameters": {}}
    adata.uns["deconvolution_m2_metadata"] = {"parameters": {}}

    with pytest.raises(ParameterError, match="Multiple deconvolution results"):
        await viz_deconv.get_deconvolution_data(adata)


@pytest.mark.asyncio
async def test_get_deconvolution_data_validates_explicit_method(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.uns["deconvolution_m1_metadata"] = {"parameters": {}}

    with pytest.raises(DataNotFoundError, match="Deconvolution 'missing' not found"):
        await viz_deconv.get_deconvolution_data(adata, method="missing")


@pytest.mark.asyncio
async def test_get_deconvolution_data_auto_select_and_context_info(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obsm["deconvolution_rctd"] = np.array([[0.8, 0.2]] * adata.n_obs)
    adata.uns["deconvolution_rctd_cell_types"] = ["T", "B"]
    adata.obs["dominant_celltype_rctd"] = ["T"] * adata.n_obs
    ctx = DummyCtx()

    out = await viz_deconv.get_deconvolution_data(adata, method=None, context=ctx)

    assert out.method == "rctd"
    assert out.proportions_key == "deconvolution_rctd"
    assert out.cell_types == ["T", "B"]
    assert out.dominant_type_key == "dominant_celltype_rctd"
    assert any("Auto-selected deconvolution method: rctd" in m for m in ctx.infos)


@pytest.mark.asyncio
async def test_get_deconvolution_data_reads_metadata_keys(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obsm["custom_props"] = np.array([[0.6, 0.4]] * adata.n_obs)
    adata.obs["custom_dom"] = ["A"] * adata.n_obs
    _add_deconv_metadata(
        adata,
        "mock",
        proportions_key="custom_props",
        cell_types=["A", "B"],
        dominant_type_key="custom_dom",
    )

    out = await viz_deconv.get_deconvolution_data(adata, method="mock")
    assert out.proportions_key == "custom_props"
    assert out.cell_types == ["A", "B"]
    assert out.dominant_type_key == "custom_dom"
    assert list(out.proportions.columns) == ["A", "B"]


@pytest.mark.asyncio
async def test_get_deconvolution_data_errors_when_proportions_key_missing(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    _add_deconv_metadata(
        adata,
        "mock",
        proportions_key="missing_props",
        cell_types=["A", "B"],
    )

    with pytest.raises(DataNotFoundError, match="Proportions data 'missing_props' not found"):
        await viz_deconv.get_deconvolution_data(adata, method="mock")


@pytest.mark.asyncio
async def test_get_deconvolution_data_fallbacks_to_generic_cell_types_with_warning(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["deconvolution_fallback"] = np.array([[0.5, 0.3, 0.2]] * adata.n_obs)
    adata.uns["deconvolution_fallback_metadata"] = {"parameters": {}}
    ctx = DummyCtx()

    out = await viz_deconv.get_deconvolution_data(adata, method="fallback", context=ctx)

    assert out.cell_types == ["CellType_0", "CellType_1", "CellType_2"]
    assert out.dominant_type_key is None
    assert any("Cell type names not found" in m for m in ctx.warnings)


@pytest.mark.asyncio
async def test_create_deconvolution_visualization_routes_aliases(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    params = VisualizationParameters(plot_type="deconvolution", subtype="dominant")

    sentinel = object()
    calls: dict[str, bool] = {}

    async def fake_dominant(*_args, **_kwargs):
        calls["dominant"] = True
        return sentinel

    monkeypatch.setattr(viz_deconv, "_create_dominant_celltype_map", fake_dominant)

    out = await viz_deconv.create_deconvolution_visualization(adata, params)
    assert out is sentinel
    assert calls["dominant"] is True


@pytest.mark.asyncio
async def test_create_deconvolution_visualization_unknown_subtype_error(
    minimal_spatial_adata,
):
    with pytest.raises(ParameterError, match="Unknown deconvolution visualization type"):
        await viz_deconv.create_deconvolution_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="deconvolution", subtype="mystery"),
        )


@pytest.mark.asyncio
async def test_create_deconvolution_visualization_routes_all_subtypes(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    sentinel = object()

    async def _spatial(*_args, **_kwargs):
        return sentinel

    async def _dominant(*_args, **_kwargs):
        return sentinel

    async def _diversity(*_args, **_kwargs):
        return sentinel

    async def _pie(*_args, **_kwargs):
        return sentinel

    async def _umap(*_args, **_kwargs):
        return sentinel

    async def _imp(*_args, **_kwargs):
        return sentinel

    monkeypatch.setattr(viz_deconv, "_create_spatial_multi_deconvolution", _spatial)
    monkeypatch.setattr(viz_deconv, "_create_dominant_celltype_map", _dominant)
    monkeypatch.setattr(viz_deconv, "_create_diversity_map", _diversity)
    monkeypatch.setattr(viz_deconv, "_create_scatterpie_plot", _pie)
    monkeypatch.setattr(viz_deconv, "_create_umap_proportions", _umap)
    monkeypatch.setattr(viz_deconv, "_create_card_imputation", _imp)

    for subtype in [None, "spatial_multi", "dominant_type", "diversity", "pie", "scatterpie", "umap", "imputation"]:
        out = await viz_deconv.create_deconvolution_visualization(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="deconvolution", subtype=subtype),
        )
        assert out is sentinel


@pytest.mark.asyncio
async def test_create_dominant_celltype_map_supports_mixed_and_non_mixed_modes(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    data = _mock_deconv_data(adata, method="rctd")
    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)
    monkeypatch.setattr(viz_deconv, "require_spatial_coords", lambda _a: _a.obsm["spatial"])

    fig = await viz_deconv._create_dominant_celltype_map(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="dominant_type",
            show_mixed_spots=True,
            min_proportion_threshold=0.75,
        ),
        context=None,
    )
    legend_labels = [t.get_text() for t in fig.axes[0].get_legend().texts]
    assert "Mixed" in legend_labels
    assert "Threshold: 0.75" in fig.axes[0].get_title()
    fig.clf()

    fig2 = await viz_deconv._create_dominant_celltype_map(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="dominant_type",
            show_mixed_spots=False,
        ),
        context=None,
    )
    legend_labels2 = [t.get_text() for t in fig2.axes[0].get_legend().texts]
    assert "Mixed" not in legend_labels2
    fig2.clf()


@pytest.mark.asyncio
async def test_create_diversity_map_renders_and_logs(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx()
    data = _mock_deconv_data(adata, method="cell2location")
    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)
    monkeypatch.setattr(viz_deconv, "require_spatial_coords", lambda _a: _a.obsm["spatial"])
    monkeypatch.setattr(
        viz_deconv,
        "entropy",
        lambda values_t, base=2: np.linspace(0.2, 0.8, values_t.shape[1]),
    )

    fig = await viz_deconv._create_diversity_map(
        adata,
        VisualizationParameters(plot_type="deconvolution", subtype="diversity"),
        context=ctx,
    )

    assert any("Mean entropy" in msg for msg in ctx.infos)
    assert "Diversity Map" in fig.axes[0].get_title()
    # main axis + colorbar axis
    assert len(fig.axes) >= 2
    fig.clf()


@pytest.mark.asyncio
async def test_create_scatterpie_plot_skips_zero_rows_and_adds_legend(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    data = _mock_deconv_data(adata, method="spotlight")
    data.proportions.iloc[0] = 0.0  # zero row path
    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)
    monkeypatch.setattr(viz_deconv, "require_spatial_coords", lambda _a: _a.obsm["spatial"])

    fig = await viz_deconv._create_scatterpie_plot(
        adata,
        VisualizationParameters(plot_type="deconvolution", subtype="pie", pie_scale=0.3),
        context=None,
    )

    assert len(fig.axes[0].patches) > 0
    assert fig.axes[0].get_legend() is not None
    fig.clf()


@pytest.mark.asyncio
async def test_create_umap_proportions_requires_umap(minimal_spatial_adata, monkeypatch):
    adata = minimal_spatial_adata.copy()
    data = _mock_deconv_data(adata)
    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)

    with pytest.raises(DataNotFoundError, match="UMAP coordinates not found"):
        await viz_deconv._create_umap_proportions(
            adata,
            VisualizationParameters(plot_type="deconvolution", subtype="umap"),
            context=None,
        )


@pytest.mark.asyncio
async def test_create_umap_proportions_renders_top_n_and_hides_unused_axes(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["X_umap"] = np.column_stack([np.arange(adata.n_obs), np.arange(adata.n_obs)])
    data = _mock_deconv_data(adata, method="rctd")
    calls = {"count": 0}

    class _CB:
        def set_label(self, *_args, **_kwargs):
            return None

    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)
    monkeypatch.setattr(
        viz_deconv.plt,
        "colorbar",
        lambda *_args, **_kwargs: calls.__setitem__("count", calls["count"] + 1) or _CB(),
    )

    fig = await viz_deconv._create_umap_proportions(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="umap",
            n_cell_types=2,
        ),
        context=None,
    )

    assert fig._suptitle is not None
    assert "Top 2 cell types" in fig._suptitle.get_text()
    assert calls["count"] == 2
    fig.clf()


@pytest.mark.asyncio
async def test_create_umap_proportions_hides_unused_axes_for_four_panels(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obsm["X_umap"] = np.column_stack([np.arange(adata.n_obs), np.arange(adata.n_obs)])
    base = _mock_deconv_data(adata, method="rctd")
    proportions = base.proportions.copy()
    proportions["NK"] = np.linspace(0.0, 0.6, adata.n_obs)
    data = viz_deconv.DeconvolutionData(
        proportions=proportions,
        method=base.method,
        cell_types=list(proportions.columns),
        proportions_key=base.proportions_key,
        dominant_type_key=base.dominant_type_key,
    )

    class _CB:
        def set_label(self, *_args, **_kwargs):
            return None

    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)
    monkeypatch.setattr(viz_deconv.plt, "colorbar", lambda *_a, **_k: _CB())

    fig = await viz_deconv._create_umap_proportions(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="umap",
            n_cell_types=4,
        ),
        context=None,
    )
    assert len(fig.axes) == 6
    hidden_axes = [ax for ax in fig.axes if not ax.axison]
    assert len(hidden_axes) >= 2
    fig.clf()


@pytest.mark.asyncio
async def test_create_spatial_multi_deconvolution_handles_nan_and_temp_cleanup(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    data = _mock_deconv_data(adata, method="card")
    data.proportions.iloc[0, 0] = np.nan
    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)

    fig = await viz_deconv._create_spatial_multi_deconvolution(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="spatial_multi",
            n_cell_types=2,
            show_colorbar=False,
        ),
        context=None,
    )

    assert "_deconv_viz_temp" not in adata.obs.columns
    assert len(fig.axes) >= 2
    fig.clf()


@pytest.mark.asyncio
async def test_create_spatial_multi_deconvolution_falls_back_to_bar_without_spatial(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.obsm.pop("spatial", None)
    data = _mock_deconv_data(adata, method="card")
    async def _get_data(*_args, **_kwargs):
        return data

    monkeypatch.setattr(viz_deconv, "get_deconvolution_data", _get_data)

    fig = await viz_deconv._create_spatial_multi_deconvolution(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="spatial_multi",
            n_cell_types=1,
        ),
        context=None,
    )

    assert fig.axes[0].get_xlabel() == "Spots (sorted)"
    assert fig.axes[0].get_ylabel() == "Proportion"
    fig.clf()


@pytest.mark.asyncio
async def test_create_card_imputation_requires_data(minimal_spatial_adata):
    with pytest.raises(DataNotFoundError, match="CARD imputation data not found"):
        await viz_deconv._create_card_imputation(
            minimal_spatial_adata,
            VisualizationParameters(plot_type="deconvolution", subtype="imputation"),
            context=None,
        )


@pytest.mark.asyncio
async def test_create_card_imputation_dominant_and_specific_feature_paths(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx()
    n = adata.n_obs
    adata.uns["card_imputation"] = {
        "proportions": pd.DataFrame(
            {
                "T": np.linspace(0.1, 0.9, n),
                "B": np.linspace(0.9, 0.1, n),
            }
        ),
        "coordinates": pd.DataFrame(
            {"x": np.linspace(0, 10, n), "y": np.linspace(0, 5, n)}
        ),
        "resolution_increase": 2.5,
    }
    calls = {"count": 0}

    class _CB:
        def set_label(self, *_args, **_kwargs):
            return None

    monkeypatch.setattr(
        viz_deconv.plt,
        "colorbar",
        lambda *_args, **_kwargs: calls.__setitem__("count", calls["count"] + 1) or _CB(),
    )

    fig1 = await viz_deconv._create_card_imputation(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="imputation",
            feature="dominant",
        ),
        context=ctx,
    )
    assert fig1.axes[0].get_legend() is not None
    fig1.clf()

    fig2 = await viz_deconv._create_card_imputation(
        adata,
        VisualizationParameters(
            plot_type="deconvolution",
            subtype="imputation",
            feature="T",
        ),
        context=ctx,
    )
    assert "CARD Imputation: T" in fig2.axes[0].get_title()
    assert calls["count"] == 1
    assert any("visualization created successfully" in m for m in ctx.infos)
    fig2.clf()


@pytest.mark.asyncio
async def test_create_card_imputation_defaults_to_dominant_feature(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    n = adata.n_obs
    adata.uns["card_imputation"] = {
        "proportions": pd.DataFrame(
            {
                "T": np.linspace(0.2, 0.7, n),
                "B": np.linspace(0.8, 0.3, n),
            }
        ),
        "coordinates": pd.DataFrame({"x": np.arange(n), "y": np.arange(n)}),
        "resolution_increase": 1.2,
    }

    fig = await viz_deconv._create_card_imputation(
        adata,
        VisualizationParameters(plot_type="deconvolution", subtype="imputation"),
        context=None,
    )
    assert "Dominant Cell Types" in fig.axes[0].get_title()
    fig.clf()


@pytest.mark.asyncio
async def test_create_card_imputation_missing_feature_error(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    n = adata.n_obs
    adata.uns["card_imputation"] = {
        "proportions": pd.DataFrame({"T": np.linspace(0.1, 0.9, n)}),
        "coordinates": pd.DataFrame({"x": np.arange(n), "y": np.arange(n)}),
        "resolution_increase": 1.5,
    }

    with pytest.raises(DataNotFoundError, match="Feature 'X' not found"):
        await viz_deconv._create_card_imputation(
            adata,
            VisualizationParameters(
                plot_type="deconvolution",
                subtype="imputation",
                feature="X",
            ),
            context=None,
        )
