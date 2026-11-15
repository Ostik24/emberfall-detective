"""MCP: the server, started as a real stdio subprocess, lists the registry's tools with the
same schemas and answers one call per tool.

Per tool: `schema_identical` (name, description, input schema equal to the registry's) and
`call_ok` (a representative call returns a non-error result that parses as JSON).
The watchlist call is `list` (writes need a user's confirmation; covered by unit tests).
"""

import asyncio
import json
import sys
from dataclasses import dataclass

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from concierge.config import REPO_ROOT
from concierge.tools.registry import TOOLS
from evals.gold import Split

METRICS = ("schema_identical", "call_ok")
GROUPS = ()
CALLS = {
    "search_catalog": {"query": "storm chasers follow tornadoes, a cow flies past", "limit": 3},
    "find_title": {"title": "Twistr"},
    "similar_to": {"id": "movie/664", "limit": 3},
    "get_title_details": {"id": "movie/664"},
    "watchlist": {"action": "list"},
    "external_lookup": {"description": "storm chasers film", "title_guesses": ["Twister"]},
}


@dataclass(frozen=True)
class MCPConfig:
    parent: str | None = None


CONFIGS = {"stdio": MCPConfig()}


async def _check() -> list[dict]:
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "concierge.mcp_server"], cwd=REPO_ROOT
    )
    expected = {t.name: t.schema() for t in TOOLS if not t.agent_only}
    cases = []
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        served = {t.name: t for t in (await session.list_tools()).tools}
        for name, schema in expected.items():
            tool = served.get(name)
            identical = tool is not None and (
                tool.description == schema["description"]
                and tool.input_schema == schema["input_schema"]
            )
            result = await session.call_tool(name, CALLS[name])
            text = result.content[0].text if result.content else ""
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            cases.append(
                {
                    "id": name,
                    "schema_identical": float(identical),
                    "call_ok": float(not result.is_error and parsed is not None),
                    "result_head": text[:300],
                }
            )
    return cases


def run(config_name: str, split: Split) -> dict:
    cases = asyncio.run(_check())
    return {
        "config": {"name": config_name, "transport": "stdio"},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["schema_identical"] * c["call_ok"] == 0],
    }
