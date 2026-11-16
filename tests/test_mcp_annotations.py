"""Every MCP tool must say whether it changes anything, so clients know when to ask the person first."""
import asyncio

from vidx import mcp_server

READ_ONLY = {"overview", "query", "get_segment", "peek", "check", "diff"}


def tools():
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


def test_every_tool_has_a_title_and_a_hint():
    for name, t in tools().items():
        assert t.title, name
        a = t.annotations
        assert a is not None and a.title, name
        assert a.read_only_hint is not None, name
        if not a.read_only_hint:
            assert a.destructive_hint is not None, name


def test_read_only_tools_are_exactly_the_ones_that_change_nothing():
    got = {n for n, t in tools().items() if t.annotations.read_only_hint}
    assert got == READ_ONLY


def test_render_is_marked_destructive_because_out_path_can_overwrite_a_file():
    assert tools()["render"].annotations.destructive_hint is True
