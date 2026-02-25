"""Unit tests for spatial_registration routing and MCP wrapper contracts."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from chatspatial.models.data import RegistrationParameters
from chatspatial.tools import spatial_registration as reg
from chatspatial.utils.exceptions import ParameterError, ProcessingError


class DummyCtx:
    def __init__(self, datasets: dict[str, object]):
        self.datasets = datasets

    async def get_adata(self, data_id: str):
        return self.datasets[data_id]


def test_validate_spatial_coords_raises_for_missing_spatial(minimal_spatial_adata):
    adata = minimal_spatial_adata.copy()
    del adata.obsm["spatial"]
    with pytest.raises(ParameterError, match="missing spatial coordinates"):
        reg._validate_spatial_coords([adata])


def test_register_slices_requires_at_least_two_slices(minimal_spatial_adata):
    with pytest.raises(ParameterError, match="at least 2 slices"):
        reg.register_slices([minimal_spatial_adata.copy()], RegistrationParameters())


def test_register_slices_dispatches_to_paste_and_stalign(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    calls: list[str] = []

    def _fake_paste(adata_list, params, spatial_key="spatial"):
        calls.append(f"paste:{spatial_key}")
        return adata_list

    def _fake_stalign(adata_list, params, spatial_key="spatial"):
        calls.append(f"stalign:{spatial_key}")
        return adata_list

    monkeypatch.setattr(reg, "_register_paste", _fake_paste)
    monkeypatch.setattr(reg, "_register_stalign", _fake_stalign)

    out1 = reg.register_slices([ad1, ad2], RegistrationParameters(method="paste"))
    out2 = reg.register_slices([ad1, ad2], RegistrationParameters(method="stalign"))
    assert out1[0] is ad1
    assert out2[0] is ad1
    assert calls == ["paste:spatial", "stalign:spatial"]


@pytest.mark.asyncio
async def test_register_spatial_slices_mcp_happy_path_records_metadata(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    src = minimal_spatial_adata.copy()
    tgt = minimal_spatial_adata.copy()
    captured: list[dict[str, object]] = []

    monkeypatch.setattr(reg, "require", lambda *_args, **_kwargs: None)

    def _fake_register_slices(adata_list, params):
        for i, adata in enumerate(adata_list):
            adata.obsm["spatial_registered"] = adata.obsm["spatial"] + i
        return adata_list

    monkeypatch.setattr(reg, "register_slices", _fake_register_slices)
    monkeypatch.setattr(
        reg,
        "store_analysis_metadata",
        lambda _adata, **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(reg, "export_analysis_result", lambda *_args, **_kwargs: [])

    out = await reg.register_spatial_slices_mcp(
        "src",
        "tgt",
        DummyCtx({"src": src, "tgt": tgt}),
        method="paste",
    )
    assert out["registration_completed"] is True
    assert out["spatial_key_registered"] == "spatial_registered"
    assert len(captured) == 2
    assert captured[0]["analysis_name"] == "registration_paste"
    assert captured[0]["results_keys"] == {"obsm": ["spatial_registered"]}


@pytest.mark.asyncio
async def test_register_spatial_slices_mcp_wraps_runtime_errors(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    src = minimal_spatial_adata.copy()
    tgt = minimal_spatial_adata.copy()
    monkeypatch.setattr(reg, "require", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        reg,
        "register_slices",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(ProcessingError, match="Registration failed: boom"):
        await reg.register_spatial_slices_mcp(
            "src",
            "tgt",
            DummyCtx({"src": src, "tgt": tgt}),
            method="paste",
        )


def test_get_common_genes_handles_duplicate_gene_names(minimal_spatial_adata):
    ad1 = minimal_spatial_adata[:, :4].copy()
    ad2 = minimal_spatial_adata[:, 2:6].copy()

    ad1.var_names = ["g0", "g0", "g2", "g3"]
    ad2.var_names = ["g2", "g3", "g4", "g4"]

    common = reg._get_common_genes([ad1, ad2])

    assert set(common) == {"g2", "g3"}


def test_transform_coordinates_handles_zero_rows_without_nan():
    transport = np.array([[0.0, 0.0], [0.2, 0.8]], dtype=float)
    ref = np.array([[1.0, 1.0], [3.0, 5.0]], dtype=float)

    out = reg._transform_coordinates(transport, ref)

    assert out.shape == (2, 2)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[1], np.array([2.6, 4.2]))


def test_register_slices_unknown_method_raises_parameter_error(minimal_spatial_adata):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    params = RegistrationParameters(method="paste").model_copy(update={"method": "unknown"})

    with pytest.raises(ParameterError, match="Unknown method"):
        reg.register_slices([ad1, ad2], params)


def test_register_stalign_rejects_non_pairwise_input(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    import sys
    import types

    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    ad3 = minimal_spatial_adata.copy()

    fake_st = types.ModuleType("STalign.STalign")
    pkg = types.ModuleType("STalign")
    pkg.STalign = fake_st
    monkeypatch.setitem(sys.modules, "STalign", pkg)
    monkeypatch.setitem(sys.modules, "STalign.STalign", fake_st)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(float32="float32", tensor=lambda x, dtype=None: x),
    )

    with pytest.raises(ParameterError, match="only supports pairwise registration"):
        reg._register_stalign([ad1, ad2, ad3], RegistrationParameters(method="stalign"))


def test_register_stalign_invalid_transform_payload_is_wrapped(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()

    monkeypatch.setattr(
        reg,
        "_prepare_stalign_image",
        lambda *_a, **_k: ([0, 1], "img"),
    )
    monkeypatch.setattr(reg, "get_device", lambda prefer_gpu=False: "cpu")

    fake_torch = types.SimpleNamespace(float32="float32", tensor=lambda x, dtype=None: np.asarray(x), Tensor=np.ndarray)

    fake_st = types.ModuleType("STalign.STalign")
    fake_st.LDDMM = lambda **_kwargs: {"A": None, "v": None, "xv": None}
    fake_st.transform_points_source_to_target = lambda xv, v, A, points: points

    pkg = types.ModuleType("STalign")
    pkg.STalign = fake_st

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "STalign", pkg)
    monkeypatch.setitem(sys.modules, "STalign.STalign", fake_st)

    with pytest.raises(ProcessingError, match="STalign registration failed"):
        reg._register_stalign([ad1, ad2], RegistrationParameters(method="stalign"))


def test_prepare_stalign_image_returns_normalized_tensor(monkeypatch: pytest.MonkeyPatch):
    fake_torch = types.SimpleNamespace(
        float32="float32",
        linspace=lambda start, stop, steps, dtype=None: np.linspace(start, stop, steps),
        tensor=lambda x, dtype=None: np.asarray(x, dtype=np.float32),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    coords = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]], dtype=np.float32)
    intensity = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    xgrid, image = reg._prepare_stalign_image(coords, intensity, (16, 12))

    assert len(xgrid) == 2
    assert image.shape == (16, 12)
    assert float(image.max()) <= 1.0
    assert float(image.min()) >= 0.0


def test_register_paste_pairwise_populates_registered_coords(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()

    def _pairwise_align(_slice1, _slice2, **_kwargs):
        return np.eye(_slice1.n_obs)

    def _stack_pairwise(slices, _pis):
        out0 = slices[0].copy()
        out1 = slices[1].copy()
        out0.obsm["spatial"] = out0.obsm["spatial"] + 1.0
        out1.obsm["spatial"] = out1.obsm["spatial"] + 2.0
        return [out0, out1]

    fake_paste = types.ModuleType("paste")
    fake_paste.pairwise_align = _pairwise_align
    fake_paste.stack_slices_pairwise = _stack_pairwise

    fake_scanpy = types.ModuleType("scanpy")
    fake_scanpy.pp = types.SimpleNamespace(
        normalize_total=lambda *_args, **_kwargs: None,
        log1p=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "paste", fake_paste)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    out = reg._register_paste([ad1, ad2], RegistrationParameters(method="paste"))
    np.testing.assert_allclose(out[0].obsm["spatial_registered"], ad1.obsm["spatial"] + 1.0)
    np.testing.assert_allclose(out[1].obsm["spatial_registered"], ad2.obsm["spatial"] + 2.0)


def test_register_paste_multi_slice_uses_center_alignment(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()
    ad3 = minimal_spatial_adata.copy()

    pairwise_calls: list[dict[str, object]] = []

    def _pairwise_align(ref_slice, moving_slice, **kwargs):
        pairwise_calls.append({"ref": ref_slice.n_obs, "moving": moving_slice.n_obs, **kwargs})
        return np.eye(moving_slice.n_obs)

    def _center_align(_ref, slices, **_kwargs):
        identity = np.eye(slices[0].n_obs)
        shift = np.full((slices[1].n_obs, slices[0].n_obs), 1.0 / slices[0].n_obs)
        return None, [identity, shift, shift]

    fake_paste = types.ModuleType("paste")
    fake_paste.pairwise_align = _pairwise_align
    fake_paste.center_align = _center_align

    fake_scanpy = types.ModuleType("scanpy")
    fake_scanpy.pp = types.SimpleNamespace(
        normalize_total=lambda *_args, **_kwargs: None,
        log1p=lambda *_args, **_kwargs: None,
    )

    monkeypatch.setitem(sys.modules, "paste", fake_paste)
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)
    monkeypatch.setattr(reg, "get_ot_backend", lambda _use_gpu: "numpy")

    params = RegistrationParameters(method="paste", reference_idx=0, use_gpu=False)
    out = reg._register_paste([ad1, ad2, ad3], params)

    assert len(pairwise_calls) == 2
    assert pairwise_calls[0]["backend"] == "numpy"
    np.testing.assert_allclose(out[0].obsm["spatial_registered"], ad1.obsm["spatial"])
    assert out[1].obsm["spatial_registered"].shape == ad2.obsm["spatial"].shape
    assert out[2].obsm["spatial_registered"].shape == ad3.obsm["spatial"].shape


def test_register_stalign_success_with_uniform_intensity(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()

    class FakeTensor(np.ndarray):
        def numpy(self):
            return np.asarray(self)

    def _to_tensor(x, dtype=None):
        arr = np.asarray(x, dtype=np.float32)
        return arr.view(FakeTensor)

    fake_torch = types.SimpleNamespace(
        float32="float32",
        tensor=_to_tensor,
        Tensor=FakeTensor,
    )

    fake_st = types.ModuleType("STalign.STalign")
    fake_st.LDDMM = lambda **_kwargs: {"A": "A", "v": "v", "xv": "xv"}
    fake_st.transform_points_source_to_target = (
        lambda _xv, _v, _A, points: _to_tensor(np.asarray(points) + 3.0)
    )

    st_pkg = types.ModuleType("STalign")
    st_pkg.STalign = fake_st

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "STalign", st_pkg)
    monkeypatch.setitem(sys.modules, "STalign.STalign", fake_st)
    monkeypatch.setattr(reg, "_prepare_stalign_image", lambda *_args, **_kwargs: ([0, 1], "img"))
    monkeypatch.setattr(reg, "get_device", lambda prefer_gpu=False: "cpu")

    params = RegistrationParameters(method="stalign", stalign_use_expression=False)
    out = reg._register_stalign([ad1, ad2], params)

    np.testing.assert_allclose(out[0].obsm["spatial_registered"], ad1.obsm["spatial"] + 3.0)
    np.testing.assert_allclose(out[1].obsm["spatial_registered"], ad2.obsm["spatial"])


def test_register_slices_defaults_to_paste_when_params_none(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    ad1 = minimal_spatial_adata.copy()
    ad2 = minimal_spatial_adata.copy()

    called = {"method": None}

    def _fake_paste(adata_list, params, spatial_key="spatial"):
        called["method"] = params.method
        return adata_list

    monkeypatch.setattr(reg, "_register_paste", _fake_paste)
    out = reg.register_slices([ad1, ad2], params=None)

    assert called["method"] == "paste"
    assert len(out) == 2


@pytest.mark.asyncio
async def test_register_spatial_slices_mcp_stalign_dependency_branch(
    minimal_spatial_adata, monkeypatch: pytest.MonkeyPatch
):
    src = minimal_spatial_adata.copy()
    tgt = minimal_spatial_adata.copy()
    requires: list[str] = []

    def _require(name, *_args, **_kwargs):
        requires.append(name)

    monkeypatch.setattr(reg, "require", _require)
    monkeypatch.setattr(
        reg,
        "register_slices",
        lambda adata_list, _params: adata_list,
    )
    monkeypatch.setattr(reg, "store_analysis_metadata", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(reg, "export_analysis_result", lambda *_args, **_kwargs: None)

    out = await reg.register_spatial_slices_mcp(
        "src", "tgt", DummyCtx({"src": src, "tgt": tgt}), method="stalign"
    )

    assert out["method"] == "stalign"
    assert requires == ["STalign"]
