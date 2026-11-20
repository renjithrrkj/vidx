"""Idle model unloading: a long-lived MCP server must not hold the speech and vision models forever."""
from vidx import idle
from vidx.analyzers import transcript, visual


def _fill_caches():
    transcript._models["small"] = object()
    visual._model = ("model", "pre", "tok", "cpu", None)


def test_unload_clears_both_model_caches():
    _fill_caches()
    assert idle.unload_models() is True
    assert transcript._models == {}
    assert visual._model is None


def test_unload_reports_nothing_to_do():
    transcript._models.clear()
    visual._model = None
    assert idle.unload_models() is False


def test_idle_tracker_unloads_only_after_quiet_period():
    t = [1000.0]
    tr = idle.Tracker(after_s=600, clock=lambda: t[0])
    _fill_caches()
    tr.touch()
    t[0] += 599
    assert tr.maybe_unload() is False
    assert visual._model is not None
    t[0] += 2
    assert tr.maybe_unload() is True
    assert visual._model is None and transcript._models == {}


def test_idle_tracker_never_unloads_while_a_tool_runs():
    t = [0.0]
    tr = idle.Tracker(after_s=600, clock=lambda: t[0])
    _fill_caches()
    with tr.busy():
        t[0] += 10_000
        assert tr.maybe_unload() is False
    assert visual._model is not None
    # the end of the call counts as activity, so the quiet period restarts
    t[0] += 599
    assert tr.maybe_unload() is False
    t[0] += 2
    assert tr.maybe_unload() is True


def test_idle_unload_can_be_turned_off():
    t = [0.0]
    tr = idle.Tracker(after_s=0, clock=lambda: t[0])
    _fill_caches()
    t[0] += 10**9
    assert tr.maybe_unload() is False
    assert visual._model is not None
    visual._model = None
    transcript._models.clear()


def test_server_reports_its_version():
    from importlib.metadata import PackageNotFoundError, version
    from vidx import mcp_server
    try:
        expected = version("vidx")
    except PackageNotFoundError:  # CI runs the tests without installing vidx
        expected = "dev"
    assert mcp_server.mcp.version == expected != ""


def test_cli_has_an_mcp_command_that_starts_the_server(monkeypatch):
    """The MCP Registry runs `uvx vidx mcp`, so the CLI must start the server."""
    import pytest
    pytest.importorskip("typer")
    from typer.testing import CliRunner
    from vidx import cli, mcp_server
    called = []
    monkeypatch.setattr(mcp_server, "main", lambda: called.append(True))
    result = CliRunner().invoke(cli.app, ["mcp"])
    assert result.exit_code == 0, result.output
    assert called == [True]
