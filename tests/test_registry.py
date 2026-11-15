"""The agent and the MCP server expose the same tools with identical schemas."""

import asyncio

from concierge import mcp_server
from concierge.tools.registry import TOOLS


def test_mcp_lists_every_data_tool_with_the_agent_schema():
    agent = {t.name: t.schema() for t in TOOLS if not t.agent_only}
    served = {t.name: t for t in mcp_server.mcp_tools()}
    assert served.keys() == agent.keys()
    for name, tool in served.items():
        assert tool.description == agent[name]["description"]
        assert tool.input_schema == agent[name]["input_schema"]


def test_respond_is_agent_only_and_the_set_has_seven_tools():
    assert len(TOOLS) == 7
    assert [t.name for t in TOOLS if t.agent_only] == ["respond"]


def test_only_watchlist_is_marked_as_writing():
    hints = {t.name: t.annotations for t in mcp_server.mcp_tools()}
    assert [n for n, a in hints.items() if not a.read_only_hint] == ["watchlist"]
    assert hints["watchlist"].destructive_hint


def test_unknown_or_agent_only_tool_is_an_error():
    for name in ("respond", "nope"):
        params = mcp_server.types.CallToolRequestParams(name=name, arguments={})
        result = asyncio.run(mcp_server.call_tool(None, params))
        assert result.is_error
