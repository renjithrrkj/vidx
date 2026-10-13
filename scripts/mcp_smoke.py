"""Smoke-test the MCP server over real stdio: list tools, call a few, show output sizes."""
import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main(video_id: str):
    params = StdioServerParameters(command="uv", args=["run", "--directory", ".", "vidx-mcp"])
    async with stdio_client(params, errlog=open("/dev/null", "w")) as (r, w):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print("tools:", [t.name for t in tools])
            for name, args in [
                ("overview", {"video_id": video_id, "limit": 6}),
                ("query", {"video_id": video_id, "text": "smartphone", "kind": "speech", "limit": 2}),
                ("get_segment", {"video_id": video_id, "start": "s99"}),
                ("edit", {"video_id": video_id, "dsl": "drop silence>1.0\nsnap words\npad 0.1\nfrobnicate"}),
                ("peek", {"target": video_id, "times": ["30", "s1"], "size": 192}),
                ("check", {"version_id": "v3"}),
            ]:
                res = await s.call_tool(name, args)
                parts = []
                for c in res.content:
                    if c.type == "text":
                        parts.append(c.text)
                    else:
                        parts.append(f"<{c.type} {c.mime_type} {len(c.data)} b64 chars>")
                print(f"--- {name} (isError={res.is_error})\n" + "\n".join(parts))


asyncio.run(main(sys.argv[1]))
