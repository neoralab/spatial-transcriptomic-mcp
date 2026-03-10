"""Contracts for async pipeline tools exposed by server layer."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from chatspatial import server as server_mod


def test_run_spatial_pipeline_sync_returns_widget_payload(monkeypatch):
    async def _fake_embeddings(**_kwargs):
        return {"ok": True}

    async def _fake_markers(**_kwargs):
        return SimpleNamespace(model_dump=lambda: {"markers": ["G1", "G2"]})

    async def _fake_visualize(**_kwargs):
        return "Visualization saved: /tmp/fake.png\nType: feature"

    monkeypatch.setattr(server_mod, "compute_embeddings", _fake_embeddings)
    monkeypatch.setattr(server_mod, "find_markers", _fake_markers)
    monkeypatch.setattr(server_mod, "visualize_data", _fake_visualize)
    monkeypatch.setattr(
        server_mod,
        "_encode_image_to_data_uri",
        lambda _path: ("data:image/png;base64,ZmFrZQ==", "image/png"),
    )

    result = asyncio.run(server_mod.run_spatial_pipeline(data_id="d1", async_mode=False))

    assert result["structuredContent"]["status"] == "completed"
    assert result["_meta"]["openai/widgetAccessible"] is True
    assert result["_meta"]["openai/widgetPrefersBorder"] is True
    assert result["_meta"]["chatspatial/widget"]["pipeline"]["type"] == "spatial_pipeline"


def test_get_pipeline_status_rejects_unknown_job_id():
    with pytest.raises(Exception):
        asyncio.run(server_mod.get_pipeline_status("missing-job"))


def test_get_pipeline_status_has_widget_meta():
    server_mod._PIPELINE_JOBS["j1"] = server_mod.PipelineJob(job_id="j1", data_id="d1", status="running")
    result = asyncio.run(server_mod.get_pipeline_status("j1"))
    assert result["_meta"]["openai/outputTemplate"] == "ui://chatspatial/widgets/pipeline-status.html"
    assert result["_meta"]["chatspatial/widget"]["pipeline_status"]["job_id"] == "j1"
