"""Unit contracts for ChatSpatial CLI entrypoint behavior."""

from __future__ import annotations

from types import SimpleNamespace

from click.testing import CliRunner

from chatspatial import __main__ as main_mod


def test_server_command_sets_mcp_settings_and_runs_transport(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run(*, transport: str):
        calls["transport"] = transport

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None, log_level=None),
        run=_fake_run,
    )
    monkeypatch.setattr(main_mod, "mcp", fake_mcp)
    monkeypatch.setattr(main_mod.config, "init_runtime", lambda **_kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        main_mod.cli,
        [
            "server",
            "--transport",
            "sse",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--log-level",
            "DEBUG",
        ],
    )

    assert result.exit_code == 0
    assert fake_mcp.settings.host == "0.0.0.0"
    assert fake_mcp.settings.port == 9000
    assert fake_mcp.settings.log_level == "DEBUG"
    assert calls["transport"] == "sse"


def test_server_command_verbose_reinitializes_runtime(monkeypatch):
    called: dict[str, object] = {}

    def _fake_init_runtime(**kwargs):
        called["kwargs"] = kwargs

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None, log_level=None),
        run=lambda **_kwargs: None,
    )
    monkeypatch.setattr(main_mod, "mcp", fake_mcp)
    monkeypatch.setattr(main_mod.config, "init_runtime", _fake_init_runtime)

    result = CliRunner().invoke(main_mod.cli, ["server", "--verbose"])
    assert result.exit_code == 0
    assert called["kwargs"] == {"verbose": True}


def test_server_command_failure_path_returns_exit_code_one(monkeypatch):
    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None, log_level=None),
        run=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("run failed")),
    )
    monkeypatch.setattr(main_mod, "mcp", fake_mcp)
    monkeypatch.setattr(main_mod.config, "init_runtime", lambda **_kwargs: None)

    result = CliRunner().invoke(main_mod.cli, ["server", "--transport", "stdio"])
    assert result.exit_code == 1
    assert "Error starting MCP server: run failed" in result.output


def test_server_command_accepts_streamable_http_transport(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run(*, transport: str):
        calls["transport"] = transport

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None, log_level=None),
        run=_fake_run,
    )
    monkeypatch.setattr(main_mod, "mcp", fake_mcp)
    monkeypatch.setattr(main_mod.config, "init_runtime", lambda **_kwargs: None)

    result = CliRunner().invoke(
        main_mod.cli,
        ["server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"],
    )

    assert result.exit_code == 0
    assert fake_mcp.settings.host == "0.0.0.0"
    assert fake_mcp.settings.port == 8080
    assert calls["transport"] == "streamable-http"


def test_server_command_cloud_run_overrides_transport_host_and_port(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run(*, transport: str):
        calls["transport"] = transport

    fake_mcp = SimpleNamespace(
        settings=SimpleNamespace(host=None, port=None, log_level=None),
        run=_fake_run,
    )
    monkeypatch.setattr(main_mod, "mcp", fake_mcp)
    monkeypatch.setattr(main_mod.config, "init_runtime", lambda **_kwargs: None)
    monkeypatch.setenv("PORT", "8787")

    result = CliRunner().invoke(
        main_mod.cli,
        ["server", "--cloud-run", "--transport", "stdio", "--host", "127.0.0.1", "--port", "9000"],
    )

    assert result.exit_code == 0
    assert fake_mcp.settings.host == "0.0.0.0"
    assert fake_mcp.settings.port == 8787
    assert calls["transport"] == "streamable-http"


def test_main_delegates_to_click_group(monkeypatch):
    called = {"value": False}

    def _fake_cli():
        called["value"] = True

    monkeypatch.setattr(main_mod, "cli", _fake_cli)
    main_mod.main()
    assert called["value"]

