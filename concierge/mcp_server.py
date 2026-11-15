"""MCP server: the six data tools of the registry over stdio, for Claude Desktop / Claude Code.

    claude mcp add emberfall -- uv --directory <repo> run python -m concierge.mcp_server

Purpose: the same tools as the notebook agent, with no logic of its own. Names, descriptions
and input schemas come straight from `concierge.tools.registry.TOOLS` (identical by
construction, checked in tests/test_registry.py); calls go to the same `Toolbox.run`.
`respond` is agent-only and not exposed. Read tools carry `read_only_hint`; `watchlist` is
marked as writing, so the MCP host asks the user before running it - and the tool itself
still only proposes add/remove until `confirm` comes with the token.
Uses the low-level `Server` of the official `mcp` SDK 2.x (FastMCP was renamed MCPServer in
2.x and derives schemas from function signatures; serving the registry's schemas unchanged
needs the low-level API).
Input / output: MCP JSON-RPC on stdin/stdout; each tool result is its pydantic model as JSON.
"""

import asyncio
import json
from functools import cache

import mcp_types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

from concierge.tools.registry import TOOLS, TOOLS_BY_NAME, Toolbox

SERVER_NAME = "emberfall-detective"


def mcp_tools() -> list[types.Tool]:
    """The registry's data tools as MCP tool definitions."""
    tools = []
    for spec in TOOLS:
        if spec.agent_only:
            continue
        schema = spec.schema()
        tools.append(
            types.Tool(
                name=schema["name"],
                description=schema["description"],
                input_schema=schema["input_schema"],
                annotations=types.ToolAnnotations(
                    read_only_hint=not spec.writes,
                    destructive_hint=spec.writes,
                    open_world_hint=spec.name == "external_lookup",
                ),
            )
        )
    return tools


@cache
def toolbox() -> Toolbox:
    """Loaded on the first call (catalog, E3, caches), then reused."""
    return Toolbox.create()


async def list_tools(ctx, params) -> types.ListToolsResult:
    return types.ListToolsResult(tools=mcp_tools())


async def call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    spec = TOOLS_BY_NAME.get(params.name)
    if spec is None or spec.agent_only:
        return _text(f"unknown tool {params.name!r}", error=True)
    try:
        output = await asyncio.to_thread(toolbox().run, params.name, params.arguments or {})
    except Exception as e:  # reported to the client as a tool error, like the agent does
        return _text(f"{type(e).__name__}: {str(e)[:300]}", error=True)
    return _text(json.dumps(output.model_dump(exclude_none=True), ensure_ascii=False))


def _text(text: str, error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=error)


server = Server(SERVER_NAME, version="0.1", on_list_tools=list_tools, on_call_tool=call_tool)


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
