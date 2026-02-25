"""Unit tests for lightweight helpers in spatial_domains and spatial_genes."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import pytest

from chatspatial.models.data import SpatialDomainParameters
from chatspatial.tools import spatial_domains as sd
from chatspatial.tools import spatial_genes as sg
from chatspatial.tools.spatial_domains import _refine_spatial_domains
from chatspatial.tools.spatial_genes import _calculate_sparse_gene_stats
from chatspatial.utils.exceptions import ParameterError, ProcessingError


class DummyCtx:
    def __init__(self, adata):
        self.adata = adata
        self.warnings: list[str] = []

    async def get_adata(self, _data_id: str):
        return self.adata

    async def warning(self, msg: str):
        self.warnings.append(msg)


def test_calculate_sparse_gene_stats_handles_dense_and_sparse_equivalently(
    minimal_spatial_adata,
):
    X_dense = np.asarray(minimal_spatial_adata.X)
    X_sparse = sp.csr_matrix(X_dense)

    dense_totals, dense_expr = _calculate_sparse_gene_stats(X_dense)
    sparse_totals, sparse_expr = _calculate_sparse_gene_stats(X_sparse)

    np.testing.assert_allclose(dense_totals, sparse_totals)
    np.testing.assert_array_equal(dense_expr, sparse_expr)


def test_refine_spatial_domains_returns_same_length_series(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["domain"] = ["0"] * 20 + ["1"] * 20 + ["2"] * 20

    refined = _refine_spatial_domains(adata, "domain", threshold=0.5)
    assert len(refined) == adata.n_obs
    assert set(refined.unique()).issubset({"0", "1", "2"})


@pytest.mark.asyncio
async def test_identify_spatial_domains_missing_spatial_coordinates_raises_processing_error(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    del adata.obsm["spatial"]
    params = SpatialDomainParameters(method="leiden", refine_domains=False)

    with pytest.raises(ProcessingError, match="No spatial coordinates found"):
        await sd.identify_spatial_domains("d1", DummyCtx(adata), params)


@pytest.mark.asyncio
async def test_identify_spatial_domains_leiden_happy_path_stores_metadata(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)
    captured: dict[str, object] = {}

    async def _fake_clustering(adata_subset, params, _ctx):
        adata_subset.obsm["X_fake"] = np.zeros((adata_subset.n_obs, 2))
        labels = ["0"] * (adata_subset.n_obs // 2) + ["1"] * (
            adata_subset.n_obs - adata_subset.n_obs // 2
        )
        return labels, "X_fake", {"silhouette": 0.5}

    monkeypatch.setattr(sd, "_identify_domains_clustering", _fake_clustering)
    monkeypatch.setattr(
        sd,
        "store_analysis_metadata",
        lambda _adata, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(sd, "export_analysis_result", lambda *_args, **_kwargs: [])

    out = await sd.identify_spatial_domains(
        "d1",
        ctx,
        SpatialDomainParameters(method="leiden", refine_domains=False),
    )
    assert out.method == "leiden"
    assert out.domain_key == "spatial_domains_leiden"
    assert out.refined_domain_key is None
    assert captured["analysis_name"] == "spatial_domains_leiden"
    assert captured["results_keys"] == {
        "obs": ["spatial_domains_leiden"],
        "obsm": ["X_fake"],
    }


@pytest.mark.asyncio
async def test_identify_spatial_domains_refinement_failure_does_not_abort(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)

    async def _fake_clustering(adata_subset, params, _ctx):
        labels = ["0"] * (adata_subset.n_obs // 2) + ["1"] * (
            adata_subset.n_obs - adata_subset.n_obs // 2
        )
        return labels, None, {"silhouette": 0.4}

    monkeypatch.setattr(sd, "_identify_domains_clustering", _fake_clustering)
    monkeypatch.setattr(sd, "_refine_spatial_domains", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("refine boom")))
    monkeypatch.setattr(sd, "store_analysis_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(sd, "export_analysis_result", lambda *_args, **_kwargs: [])

    out = await sd.identify_spatial_domains(
        "d1",
        ctx,
        SpatialDomainParameters(method="leiden", refine_domains=True),
    )
    assert out.refined_domain_key is None
    assert any("Domain refinement failed" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_identify_spatial_genes_routes_to_spatialde(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)

    async def _fake_spatialde(data_id, adata_obj, params, _ctx):
        return sg.SpatialVariableGenesResult(
            data_id=data_id,
            method="spatialde",
            n_genes_analyzed=10,
            n_significant_genes=2,
            spatial_genes=["gene_0", "gene_1"],
            results_key="spatialde_results_d1",
        )

    monkeypatch.setattr(sg, "_identify_spatial_genes_spatialde", _fake_spatialde)

    out = await sg.identify_spatial_genes(
        "d1",
        ctx,
        sg.SpatialVariableGenesParameters(method="spatialde", spatial_key="spatial"),
    )
    assert out.method == "spatialde"
    assert out.n_significant_genes == 2


@pytest.mark.asyncio
async def test_identify_spatial_genes_routes_to_sparkx(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)

    async def _fake_sparkx(data_id, adata_obj, params, _ctx):
        return sg.SpatialVariableGenesResult(
            data_id=data_id,
            method="sparkx",
            n_genes_analyzed=8,
            n_significant_genes=1,
            spatial_genes=["gene_2"],
            results_key="sparkx_results_d1",
        )

    monkeypatch.setattr(sg, "_identify_spatial_genes_sparkx", _fake_sparkx)

    out = await sg.identify_spatial_genes(
        "d1",
        ctx,
        sg.SpatialVariableGenesParameters(method="sparkx", spatial_key="spatial"),
    )
    assert out.method == "sparkx"
    assert out.spatial_genes == ["gene_2"]


@pytest.mark.asyncio
async def test_identify_spatial_genes_routes_to_flashs(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)

    async def _fake_flashs(data_id, adata_obj, params, _ctx):
        return sg.SpatialVariableGenesResult(
            data_id=data_id,
            method="flashs",
            n_genes_analyzed=12,
            n_significant_genes=3,
            spatial_genes=["gene_1", "gene_4", "gene_8"],
            results_key="flashs_results_d1",
        )

    monkeypatch.setattr(sg, "_identify_spatial_genes_flashs", _fake_flashs)

    out = await sg.identify_spatial_genes(
        "d1",
        ctx,
        sg.SpatialVariableGenesParameters(method="flashs", spatial_key="spatial"),
    )
    assert out.method == "flashs"
    assert out.n_significant_genes == 3


@pytest.mark.asyncio
async def test_identify_spatial_genes_rejects_unknown_method_via_runtime_guard(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)
    params = sg.SpatialVariableGenesParameters(method="sparkx").model_copy(
        update={"method": "unknown_method"}
    )

    with pytest.raises(ParameterError, match="Unsupported method"):
        await sg.identify_spatial_genes("d1", ctx, params)


@pytest.mark.asyncio
async def test_identify_spatial_genes_requires_spatial_coordinates(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    del adata.obsm["spatial"]
    ctx = DummyCtx(adata)

    with pytest.raises(Exception, match="Spatial coordinates"):
        await sg.identify_spatial_genes(
            "d1", ctx, sg.SpatialVariableGenesParameters(method="sparkx")
        )


@pytest.mark.asyncio
async def test_identify_spatial_domains_unknown_method_is_wrapped(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata.copy()
    params = SpatialDomainParameters(method="leiden").model_copy(
        update={"method": "unknown"}
    )

    with pytest.raises(ProcessingError, match="Unsupported method"):
        await sd.identify_spatial_domains("d1", DummyCtx(adata), params)


@pytest.mark.asyncio
async def test_identify_domains_clustering_louvain_falls_back_to_leiden(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)
    # _identify_domains_clustering expects expression graph to exist.
    adata.obsp["connectivities"] = np.eye(adata.n_obs, dtype=float)

    monkeypatch.setattr(sd, "ensure_pca", lambda *_a, **_k: None)
    monkeypatch.setattr(sd, "ensure_neighbors", lambda *_a, **_k: None)

    class _FakeSq:
        class gr:
            @staticmethod
            def spatial_neighbors(a, coord_type="generic"):
                del coord_type
                a.obsp["spatial_connectivities"] = np.eye(a.n_obs, dtype=float)

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSq)
    monkeypatch.setattr(sd.sc.tl, "louvain", lambda *_a, **_k: (_ for _ in ()).throw(ImportError("missing louvain")))

    def _fake_leiden(a, resolution=1.0, key_added="spatial_leiden"):
        del resolution
        a.obs[key_added] = ["0"] * a.n_obs

    monkeypatch.setattr(sd.sc.tl, "leiden", _fake_leiden)

    labels, emb_key, stats = await sd._identify_domains_clustering(
        adata,
        SpatialDomainParameters(method="louvain", resolution=0.5),
        ctx,
    )

    assert len(labels) == adata.n_obs
    assert emb_key == "X_pca"
    assert stats["method"] == "louvain"
    assert any("deprecated" in w.lower() for w in ctx.warnings)
    assert any("using leiden clustering instead" in w.lower() for w in ctx.warnings)


@pytest.mark.asyncio
async def test_identify_domains_clustering_spatial_graph_failure_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    monkeypatch.setattr(sd, "ensure_pca", lambda *_a, **_k: None)
    monkeypatch.setattr(sd, "ensure_neighbors", lambda *_a, **_k: None)

    class _FakeSq:
        class gr:
            @staticmethod
            def spatial_neighbors(_a, coord_type="generic"):
                del coord_type
                raise RuntimeError("sq boom")

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSq)

    with pytest.raises(ProcessingError, match="Spatial graph construction failed"):
        await sd._identify_domains_clustering(
            adata,
            SpatialDomainParameters(method="leiden"),
            DummyCtx(adata),
        )


def test_refine_spatial_domains_single_spot_raises_processing_error(
    minimal_spatial_adata,
):
    adata = minimal_spatial_adata[:1].copy()
    adata.obs["domain"] = ["A"]

    with pytest.raises(ProcessingError, match="All spatial coordinates are identical"):
        _refine_spatial_domains(adata, "domain", threshold=0.5)


@pytest.mark.asyncio
async def test_identify_spatial_domains_cleans_dense_nan_inf_before_clustering(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    dense = np.asarray(adata.X, dtype=float)
    dense[0, 0] = np.nan
    dense[1, 1] = np.inf
    adata.X = dense

    async def _fake_clustering(adata_subset, params, _ctx):
        del params
        assert np.isfinite(np.asarray(adata_subset.X)).all()
        return ["0"] * adata_subset.n_obs, None, {"ok": 1}

    monkeypatch.setattr(sd, "_identify_domains_clustering", _fake_clustering)
    monkeypatch.setattr(sd, "store_analysis_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(sd, "export_analysis_result", lambda *_a, **_k: [])

    ctx = DummyCtx(adata)
    out = await sd.identify_spatial_domains(
        "d1", ctx, SpatialDomainParameters(method="leiden", refine_domains=False)
    )

    assert out.n_domains == 1
    assert any("NaN or infinite values" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_identify_spatial_domains_uses_raw_when_current_has_negatives(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    adata.raw = adata.copy()
    adata.X = np.asarray(adata.X, dtype=float)
    adata.X[0, 0] = -1.0

    async def _fake_clustering(adata_subset, params, _ctx):
        del params
        assert np.min(np.asarray(adata_subset.X)) >= 0.0
        return ["0"] * adata_subset.n_obs, None, {"ok": 1}

    monkeypatch.setattr(sd, "_identify_domains_clustering", _fake_clustering)
    monkeypatch.setattr(sd, "store_analysis_metadata", lambda *_a, **_k: None)
    monkeypatch.setattr(sd, "export_analysis_result", lambda *_a, **_k: [])

    ctx = DummyCtx(adata)
    await sd.identify_spatial_domains(
        "d1", ctx, SpatialDomainParameters(method="leiden", refine_domains=False)
    )

    assert any("negative values" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_identify_domains_spagcn_success_with_dummy_histology_fallback(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()
    ctx = DummyCtx(adata)

    class _FakeSpg:
        @staticmethod
        def prefilter_genes(_adata, min_cells=3):
            del _adata, min_cells

        @staticmethod
        def prefilter_specialgenes(_adata):
            del _adata

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSpg)
    monkeypatch.setattr("chatspatial.utils.compat.ensure_spagcn_compat", lambda *_a, **_k: None)

    import sys
    import types

    ez_mod = types.ModuleType("SpaGCN.ez_mode")
    ez_mod.detect_spatial_domains_ez_mode = lambda ad, *args, **kwargs: ["0"] * ad.n_obs
    pkg = types.ModuleType("SpaGCN")
    pkg.ez_mode = ez_mod
    sys.modules["SpaGCN"] = pkg
    sys.modules["SpaGCN.ez_mode"] = ez_mod

    params = SpatialDomainParameters(
        method="spagcn",
        n_domains=20,
        spagcn_use_histology=True,
    )

    labels, emb_key, stats = await sd._identify_domains_spagcn(adata, params, ctx)

    assert len(labels) == adata.n_obs
    assert emb_key is None
    assert stats["method"] == "spagcn"
    assert stats["use_histology"] is False
    assert any("unstable or noisy" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_identify_domains_spagcn_timeout_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    class _FakeSpg:
        @staticmethod
        def prefilter_genes(_adata, min_cells=3):
            del _adata, min_cells

        @staticmethod
        def prefilter_specialgenes(_adata):
            del _adata

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSpg)
    monkeypatch.setattr("chatspatial.utils.compat.ensure_spagcn_compat", lambda *_a, **_k: None)

    import sys
    import types
    import asyncio as _asyncio

    ez_mod = types.ModuleType("SpaGCN.ez_mode")
    ez_mod.detect_spatial_domains_ez_mode = lambda ad, *args, **kwargs: ["0"] * ad.n_obs
    pkg = types.ModuleType("SpaGCN")
    pkg.ez_mode = ez_mod
    sys.modules["SpaGCN"] = pkg
    sys.modules["SpaGCN.ez_mode"] = ez_mod

    async def _raise_timeout(_future, timeout=None):
        del _future, timeout
        raise _asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.wait_for", _raise_timeout)

    with pytest.raises(ProcessingError, match="SpaGCN timed out"):
        await sd._identify_domains_spagcn(
            adata,
            SpatialDomainParameters(method="spagcn", timeout=1),
            DummyCtx(adata),
        )


@pytest.mark.asyncio
async def test_identify_domains_spagcn_rejects_mismatched_coordinate_length(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    adata = minimal_spatial_adata.copy()

    class _FakeSpg:
        @staticmethod
        def prefilter_genes(_adata, min_cells=3):
            del _adata, min_cells

        @staticmethod
        def prefilter_specialgenes(_adata):
            del _adata

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSpg)
    monkeypatch.setattr("chatspatial.utils.compat.ensure_spagcn_compat", lambda *_a, **_k: None)
    monkeypatch.setattr(sd, "require_spatial_coords", lambda _adata: np.asarray(_adata.obsm["spatial"])[:-1])

    import sys
    import types

    ez_mod = types.ModuleType("SpaGCN.ez_mode")
    ez_mod.detect_spatial_domains_ez_mode = lambda ad, *args, **kwargs: ["0"] * ad.n_obs
    pkg = types.ModuleType("SpaGCN")
    pkg.ez_mode = ez_mod
    sys.modules["SpaGCN"] = pkg
    sys.modules["SpaGCN.ez_mode"] = ez_mod

    with pytest.raises(ProcessingError, match="doesn't match data"):
        await sd._identify_domains_spagcn(
            adata,
            SpatialDomainParameters(method="spagcn"),
            DummyCtx(adata),
        )


@pytest.mark.asyncio
async def test_identify_domains_stagate_rejects_unsupported_torch_version(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.9.0"
    torch_mod.device = lambda x: x
    sys.modules["torch"] = torch_mod

    with pytest.raises(ProcessingError, match="requires PyTorch <= 2.8.0"):
        await sd._identify_domains_stagate(
            adata,
            SpatialDomainParameters(method="stagate"),
            DummyCtx(adata),
        )


@pytest.mark.asyncio
async def test_identify_domains_stagate_success_returns_embeddings_and_stats(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.8.0"
    torch_mod.device = lambda x: f"device:{x}"
    sys.modules["torch"] = torch_mod

    class _FakeSTAGATE:
        @staticmethod
        def Cal_Spatial_Net(a, rad_cutoff=50):
            a.uns["rad_cutoff"] = rad_cutoff

        @staticmethod
        def Stats_Spatial_Net(_a):
            return None

        @staticmethod
        def train_STAGATE(a, device=None):
            del device
            a.obsm["STAGATE"] = np.ones((a.n_obs, 4), dtype=float)
            return a

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSTAGATE)
    monkeypatch.setattr(
        "chatspatial.utils.compute.gmm_clustering",
        lambda data, n_clusters, **_kwargs: np.arange(data.shape[0]) % n_clusters,
    )

    async def _fake_resolve_device(prefer_gpu: bool, ctx):
        del prefer_gpu, ctx
        return "cpu"

    monkeypatch.setattr(sd, "resolve_device_async", _fake_resolve_device)

    labels, emb_key, stats = await sd._identify_domains_stagate(
        adata,
        SpatialDomainParameters(
            method="stagate",
            n_domains=3,
            stagate_use_gpu=False,
            stagate_rad_cutoff=40,
        ),
        DummyCtx(adata),
    )

    assert len(labels) == adata.n_obs
    assert emb_key == "STAGATE"
    assert "STAGATE" in adata.obsm
    assert stats["method"] == "stagate_pyg"
    assert stats["target_n_clusters"] == 3
    assert stats["rad_cutoff"] == 40


@pytest.mark.asyncio
async def test_identify_domains_stagate_timeout_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import asyncio as _asyncio
    import sys
    import types

    adata = minimal_spatial_adata.copy()

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.8.0"
    torch_mod.device = lambda x: x
    sys.modules["torch"] = torch_mod

    class _FakeSTAGATE:
        @staticmethod
        def Cal_Spatial_Net(_a, rad_cutoff=50):
            del rad_cutoff

        @staticmethod
        def train_STAGATE(a, device=None):
            del a, device
            return None

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: _FakeSTAGATE)

    async def _fake_resolve_device(prefer_gpu: bool, ctx):
        del prefer_gpu, ctx
        return "cpu"

    monkeypatch.setattr(sd, "resolve_device_async", _fake_resolve_device)

    async def _raise_timeout(_future, timeout=None):
        del _future, timeout
        raise _asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.wait_for", _raise_timeout)

    with pytest.raises(ProcessingError, match="STAGATE training timeout"):
        await sd._identify_domains_stagate(
            adata,
            SpatialDomainParameters(method="stagate", timeout=1),
            DummyCtx(adata),
        )


@pytest.mark.asyncio
async def test_identify_domains_graphst_mclust_path_success(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import warnings
    import sys
    import types

    adata = minimal_spatial_adata.copy()

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.8.0"
    torch_mod.device = lambda x: f"device:{x}"
    sys.modules["torch"] = torch_mod

    class _FakeGraphST:
        def __init__(self, adata_graphst, device=None, random_seed=0):
            del device, random_seed
            self._adata = adata_graphst

        def train(self):
            self._adata.obsm["emb"] = np.ones((self._adata.n_obs, 24), dtype=float)
            return self._adata

    graphst_sub = types.ModuleType("GraphST.GraphST")
    graphst_sub.GraphST = _FakeGraphST
    graphst_pkg = types.ModuleType("GraphST")
    graphst_pkg.GraphST = graphst_sub
    sys.modules["GraphST"] = graphst_pkg
    sys.modules["GraphST.GraphST"] = graphst_sub

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "chatspatial.utils.compute.gmm_clustering",
        lambda data, n_clusters, **_kwargs: np.arange(data.shape[0]) % n_clusters,
    )

    async def _fake_resolve_device(prefer_gpu: bool, ctx):
        del prefer_gpu, ctx
        return "cpu"

    monkeypatch.setattr(sd, "resolve_device_async", _fake_resolve_device)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        labels, emb_key, stats = await sd._identify_domains_graphst(
            adata,
            SpatialDomainParameters(
                method="graphst",
                graphst_clustering_method="mclust",
                graphst_refinement=False,
                graphst_n_clusters=4,
                n_domains=4,
            ),
            DummyCtx(adata),
        )

    assert len(labels) == adata.n_obs
    assert emb_key == "emb"
    assert "emb" in adata.obsm
    assert stats["method"] == "graphst"
    assert stats["clustering_method"] == "mclust"
    assert stats["n_clusters"] == 4


@pytest.mark.asyncio
async def test_identify_domains_graphst_timeout_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import asyncio as _asyncio
    import sys
    import types

    adata = minimal_spatial_adata.copy()

    torch_mod = types.ModuleType("torch")
    torch_mod.__version__ = "2.8.0"
    torch_mod.device = lambda x: x
    sys.modules["torch"] = torch_mod

    class _FakeGraphST:
        def __init__(self, adata_graphst, device=None, random_seed=0):
            del adata_graphst, device, random_seed

        def train(self):
            return adata

    graphst_sub = types.ModuleType("GraphST.GraphST")
    graphst_sub.GraphST = _FakeGraphST
    graphst_pkg = types.ModuleType("GraphST")
    graphst_pkg.GraphST = graphst_sub
    sys.modules["GraphST"] = graphst_pkg
    sys.modules["GraphST.GraphST"] = graphst_sub

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: None)

    async def _fake_resolve_device(prefer_gpu: bool, ctx):
        del prefer_gpu, ctx
        return "cpu"

    monkeypatch.setattr(sd, "resolve_device_async", _fake_resolve_device)

    async def _raise_timeout(_future, timeout=None):
        del _future, timeout
        raise _asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.wait_for", _raise_timeout)

    with pytest.raises(ProcessingError, match="GraphST training timeout"):
        await sd._identify_domains_graphst(
            adata,
            SpatialDomainParameters(method="graphst", timeout=1),
            DummyCtx(adata),
        )


@pytest.mark.asyncio
async def test_identify_domains_banksy_success_path(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    adata = minimal_spatial_adata.copy()
    banksy_matrix = adata.copy()

    init_mod = types.ModuleType("banksy.initialize_banksy")
    init_mod.initialize_banksy = lambda *_a, **_k: {"ok": True}
    embed_mod = types.ModuleType("banksy.embed_banksy")
    embed_mod.generate_banksy_matrix = lambda *_a, **_k: (None, banksy_matrix)

    sys.modules["banksy"] = types.ModuleType("banksy")
    sys.modules["banksy.initialize_banksy"] = init_mod
    sys.modules["banksy.embed_banksy"] = embed_mod

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: None)
    monkeypatch.setattr(
        sd.sc.pp,
        "pca",
        lambda a, n_comps=20: a.obsm.__setitem__("X_pca", np.ones((a.n_obs, n_comps))),
    )
    monkeypatch.setattr(sd.sc.pp, "neighbors", lambda *_a, **_k: None)
    monkeypatch.setattr(
        sd.sc.tl,
        "leiden",
        lambda a, resolution=0.5, key_added="banksy_cluster": a.obs.__setitem__(
            key_added, ["0"] * (a.n_obs // 2) + ["1"] * (a.n_obs - a.n_obs // 2)
        ),
    )

    labels, emb_key, stats = await sd._identify_domains_banksy(
        adata,
        SpatialDomainParameters(method="banksy", banksy_pca_dims=5, n_domains=2),
        DummyCtx(adata),
    )

    assert len(labels) == adata.n_obs
    assert emb_key == "X_banksy_pca"
    assert "X_banksy_pca" in adata.obsm
    assert stats["method"] == "banksy"
    assert stats["n_clusters"] == 2
    assert stats["feature_expansion"].endswith("x")


@pytest.mark.asyncio
async def test_identify_domains_banksy_timeout_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import asyncio as _asyncio
    import sys
    import types

    adata = minimal_spatial_adata.copy()

    init_mod = types.ModuleType("banksy.initialize_banksy")
    init_mod.initialize_banksy = lambda *_a, **_k: {"ok": True}
    embed_mod = types.ModuleType("banksy.embed_banksy")
    embed_mod.generate_banksy_matrix = lambda *_a, **_k: (None, adata.copy())

    sys.modules["banksy"] = types.ModuleType("banksy")
    sys.modules["banksy.initialize_banksy"] = init_mod
    sys.modules["banksy.embed_banksy"] = embed_mod

    monkeypatch.setattr(sd, "require", lambda *_a, **_k: None)

    async def _raise_timeout(_future, timeout=None):
        del _future, timeout
        raise _asyncio.TimeoutError()

    monkeypatch.setattr("asyncio.wait_for", _raise_timeout)

    with pytest.raises(ProcessingError, match="BANKSY timeout"):
        await sd._identify_domains_banksy(
            adata,
            SpatialDomainParameters(method="banksy", timeout=1),
            DummyCtx(adata),
        )
