"""Extended unit tests for high-impact AnnData utility paths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from chatspatial.utils import adata_utils as au
from chatspatial.utils.exceptions import DataError, DataNotFoundError, ParameterError


class _WarnCtx:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def warning(self, msg: str) -> None:
        self.messages.append(msg)


def test_sample_expression_values_handles_dense_sparse_and_empty_sparse(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()

    dense_sample = au.sample_expression_values(adata, n_samples=5)
    assert dense_sample.shape[0] == 5

    adata_sparse = adata.copy()
    adata_sparse.X = sparse.csr_matrix(np.asarray(adata_sparse.X))
    sparse_sample = au.sample_expression_values(adata_sparse, n_samples=7)
    assert sparse_sample.shape[0] <= 7

    adata_empty = adata.copy()
    adata_empty.X = sparse.csr_matrix((adata_empty.n_obs, adata_empty.n_vars), dtype=float)
    empty_sample = au.sample_expression_values(adata_empty, n_samples=3)
    assert empty_sample.shape[0] <= 3 * adata_empty.n_vars


def test_require_spatial_coords_validates_nan_inf_and_missing_key(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obsm["spatial"][0, 0] = np.nan
    with pytest.raises(DataError, match="contain NaN"):
        au.require_spatial_coords(adata)

    adata = minimal_spatial_adata.copy()
    adata.obsm["spatial"][0, 0] = np.inf
    with pytest.raises(DataError, match="infinite"):
        au.require_spatial_coords(adata)

    with pytest.raises(DataError, match="not found in adata.obsm"):
        au.require_spatial_coords(minimal_spatial_adata.copy(), spatial_key="not_here")


def test_validate_obs_and_var_column_raise_useful_errors(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()

    with pytest.raises(DataError, match="Cell type not found"):
        au.validate_obs_column(adata, "cell_type", "Cell type")

    with pytest.raises(DataError, match="Marker not found"):
        au.validate_var_column(adata, "marker", "Marker")


def test_validate_adata_basics_check_empty_ratio_for_dense_and_sparse(minimal_spatial_adata):
    dense = minimal_spatial_adata.copy()
    dense.X = np.asarray(dense.X)
    dense.X[:20, :] = 0.0
    with pytest.raises(DataError, match="cells .* zero expression"):
        au.validate_adata_basics(dense, check_empty_ratio=True, max_empty_obs_ratio=0.1)

    sparse_adata = minimal_spatial_adata.copy()
    sparse_adata.X = sparse.csr_matrix(np.asarray(sparse_adata.X))
    sparse_matrix = sparse_adata.X.tolil()
    sparse_matrix[:, :18] = 0.0
    sparse_adata.X = sparse_matrix.tocsr()
    with pytest.raises(DataError, match="genes .* zero expression"):
        au.validate_adata_basics(sparse_adata, check_empty_ratio=True, max_empty_vars_ratio=0.5)


def test_ensure_categorical_converts_and_rejects_missing_column(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["cluster"] = ["A", "B"] * (adata.n_obs // 2)

    au.ensure_categorical(adata, "cluster")
    assert pd.api.types.is_categorical_dtype(adata.obs["cluster"])

    with pytest.raises(DataError, match="Column 'missing' not found"):
        au.ensure_categorical(adata, "missing")


def test_shallow_copy_adata_shares_expression_but_copies_metadata(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.layers["counts"] = np.asarray(adata.X).copy()
    adata.uns["meta"] = {"v": 1}

    shallow = au.shallow_copy_adata(adata)

    assert shallow is not adata
    assert shallow.X is adata.X
    assert shallow.layers["counts"] is adata.layers["counts"]
    assert shallow.obs is not adata.obs
    assert shallow.var is not adata.var
    assert shallow.uns is not adata.uns


def test_store_and_reconstruct_velovi_essential_data_roundtrip(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    vel = minimal_spatial_adata[:, :6].copy()
    vel.layers["velocity_velovi"] = np.full((vel.n_obs, vel.n_vars), 0.5, dtype=np.float32)
    vel.layers["Ms"] = np.full((vel.n_obs, vel.n_vars), 1.0, dtype=np.float32)
    vel.layers["Mu"] = np.full((vel.n_obs, vel.n_vars), 2.0, dtype=np.float32)
    vel.obsp["connectivities"] = sparse.eye(vel.n_obs, format="csr")
    vel.obsp["distances"] = sparse.eye(vel.n_obs, format="csr") * 2

    au.store_velovi_essential_data(adata, vel)
    assert au.has_velovi_essential_data(adata) is True

    reconstructed = au.reconstruct_velovi_adata(adata)
    assert reconstructed.n_obs == adata.n_obs
    assert reconstructed.n_vars == vel.n_vars
    assert "velocity" in reconstructed.layers
    assert "Ms" in reconstructed.layers
    assert "Mu" in reconstructed.layers
    assert "connectivities" in reconstructed.obsp
    assert "distances" in reconstructed.obsp
    assert "neighbors" in reconstructed.uns
    assert "spatial" in reconstructed.obsm


def test_reconstruct_velovi_adata_requires_essential_keys(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    with pytest.raises(DataError, match="velovi_gene_names not found"):
        au.reconstruct_velovi_adata(adata)

    adata.uns["velovi_gene_names"] = ["g1", "g2"]
    with pytest.raises(DataError, match="velovi_velocity not found"):
        au.reconstruct_velovi_adata(adata)


def test_standardize_adata_moves_spatial_and_casts_known_categories(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obsm["coords"] = adata.obsm["spatial"].copy()
    del adata.obsm["spatial"]
    adata.obs["cell_type"] = ["T", "B"] * (adata.n_obs // 2)

    out = au.standardize_adata(adata, copy=True)

    assert "spatial" in out.obsm
    assert pd.api.types.is_categorical_dtype(out.obs["cell_type"])
    assert "spatial" not in adata.obsm


def test_validate_adata_required_keys_and_integrity_checks(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()

    with pytest.raises(DataError, match="Missing required keys"):
        au.validate_adata(adata, required_keys={"obs": ["missing_col"]})

    # Required keys pass, but detailed checks should fail with aggregated issues.
    adata.layers.pop("spliced", None)
    adata.layers.pop("unspliced", None)
    adata.obsm["spatial"] = np.ones((adata.n_obs, 2), dtype=float)
    with pytest.raises(DataError, match="Validation failed"):
        au.validate_adata(
            adata,
            required_keys={},
            check_spatial=True,
            check_velocity=True,
        )


def test_store_analysis_metadata_and_get_parameter(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()

    au.store_analysis_metadata(
        adata,
        analysis_name="spatial_stats_neighborhood",
        method="squidpy",
        parameters={"cluster_key": "leiden"},
        results_keys={"uns": ["neighborhood_enrichment"]},
        statistics={"n_clusters": 3},
        species="human",
        database="consensus",
    )

    assert "spatial_stats_neighborhood_metadata" in adata.uns
    assert (
        au.get_analysis_parameter(
            adata,
            analysis_name="spatial_stats_neighborhood",
            parameter_name="cluster_key",
            default=None,
        )
        == "leiden"
    )
    assert (
        au.get_analysis_parameter(
            adata,
            analysis_name="missing_analysis",
            parameter_name="cluster_key",
            default="fallback",
        )
        == "fallback"
    )


def test_hvg_and_gene_selection_paths(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.var["highly_variable"] = False
    adata.var.loc[adata.var_names[:5], "highly_variable"] = True

    hvgs = au.get_highly_variable_genes(adata, max_genes=3)
    assert len(hvgs) == 3

    selected = au.select_genes_for_analysis(adata, genes=["gene_0", "missing"], n_genes=2)
    assert selected == ["gene_0"]

    with pytest.raises(DataError, match="Did you mean"):
        au.select_genes_for_analysis(adata, genes=["gene_00"], n_genes=2)

    adata_no_hvg = minimal_spatial_adata.copy()
    out = au.select_genes_for_analysis(
        adata_no_hvg,
        genes=None,
        n_genes=2,
        require_hvg=False,
    )
    assert out == []


def test_hvg_fallback_to_variance_for_sparse_and_dense(minimal_spatial_adata):
    dense = minimal_spatial_adata.copy()
    dense.var = dense.var.drop(columns=[c for c in dense.var.columns if c == "highly_variable"], errors="ignore")
    dense_hvgs = au.get_highly_variable_genes(dense, max_genes=4, fallback_to_variance=True)
    assert len(dense_hvgs) == 4

    sparse_adata = minimal_spatial_adata.copy()
    sparse_adata.var = sparse_adata.var.drop(columns=[c for c in sparse_adata.var.columns if c == "highly_variable"], errors="ignore")
    sparse_adata.X = sparse.csr_matrix(np.asarray(sparse_adata.X))
    sparse_hvgs = au.get_highly_variable_genes(sparse_adata, max_genes=4, fallback_to_variance=True)
    assert len(sparse_hvgs) == 4


def test_make_unique_names_and_var_name_fixers(minimal_spatial_adata):
    assert au.make_unique_names(["A", "B", "A", "A"]) == ["A", "B", "A-1", "A-2"]

    adata = minimal_spatial_adata.copy()
    adata.var_names = [f"gene_{i // 2}" for i in range(adata.n_vars)]
    n_fixed = au.ensure_unique_var_names(adata)
    assert n_fixed > 0
    assert adata.var_names.is_unique


@pytest.mark.asyncio
async def test_ensure_unique_var_names_async_warns(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.var_names = ["dup"] * adata.n_vars
    ctx = _WarnCtx()

    n_fixed = await au.ensure_unique_var_names_async(adata, ctx, label="reference")

    assert n_fixed == adata.n_vars - 1
    assert any("duplicate gene names" in m for m in ctx.messages)


def test_check_is_integer_counts_and_ensure_counts_layer(minimal_spatial_adata):
    dense = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert au.check_is_integer_counts(dense) == (True, False, False)

    non_int = np.array([[1.2, 2.0], [3.0, -4.0]])
    is_int, has_neg, has_dec = au.check_is_integer_counts(non_int)
    assert is_int is False
    assert has_neg is True
    assert has_dec is True

    adata = minimal_spatial_adata.copy()
    adata.layers["counts"] = np.asarray(adata.X).copy()
    created = au.ensure_counts_layer(adata)
    assert created is False

    adata2 = minimal_spatial_adata.copy()
    adata2.raw = adata2.copy()
    adata2 = adata2[:, :5].copy()
    created = au.ensure_counts_layer(adata2)
    assert created is True
    assert adata2.layers["counts"].shape == (adata2.n_obs, adata2.n_vars)

    adata3 = minimal_spatial_adata.copy()
    adata3.raw = None
    adata3.layers.clear()
    with pytest.raises(DataNotFoundError, match="Cannot create 'counts' layer"):
        au.ensure_counts_layer(adata3)


def test_get_gene_and_genes_expression_paths(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.layers["counts"] = np.asarray(adata.X).copy()

    one = au.get_gene_expression(adata, "gene_0")
    assert one.shape == (adata.n_obs,)

    one_counts = au.get_gene_expression(adata, "gene_0", layer="counts")
    assert np.array_equal(one_counts, one)

    many = au.get_genes_expression(adata, ["gene_0", "gene_1"], layer="counts")
    assert many.shape == (adata.n_obs, 2)

    with pytest.raises(DataError, match="Gene 'missing' not found"):
        au.get_gene_expression(adata, "missing")

    with pytest.raises(DataError, match="Layer 'bad' not found"):
        au.get_gene_expression(adata, "gene_0", layer="bad")

    with pytest.raises(DataError, match="Genes not found"):
        au.get_genes_expression(adata, ["gene_0", "missing"])


def test_profiles_and_overlap_helpers(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    adata.obs["num_col"] = np.arange(adata.n_obs)
    adata.obs["cat_col"] = [f"v{i}" for i in range(adata.n_obs)]

    obs_profile = au.get_column_profile(adata, layer="obs")
    names = {item["name"]: item for item in obs_profile}
    assert names["num_col"]["dtype"] == "numerical"
    assert names["cat_col"]["dtype"] == "categorical"
    assert len(names["cat_col"]["sample_values"]) == 5

    profile = au.get_adata_profile(adata)
    assert "obs_columns" in profile
    assert "top_expressed_genes" in profile

    common = au.find_common_genes(["A", "B", "C"], ["B", "C"], ["C", "D"])
    assert common == ["C"]

    with pytest.raises(ParameterError, match="at least 2 gene collections"):
        au.find_common_genes(["A"])

    with pytest.raises(DataError, match="Insufficient gene overlap"):
        au.validate_gene_overlap(["A", "B"], source_n_genes=200, target_n_genes=150, min_genes=10)
