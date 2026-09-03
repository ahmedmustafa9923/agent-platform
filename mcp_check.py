import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    question = " ".join(sys.argv[1:]) or "what are the specifications"

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_rag.mcp_server"],
        env={**os.environ},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await session.list_tools()
            print("tools the server offers:", [t.name for t in tools.tools])

            result = await session.call_tool("search_catalog", {"query": question, "top_k": 2})
            payload = getattr(result, "structured_content", None)
            if payload is None:
                payload = json.loads(result.content[0].text)

            print("\nmatch_count:", payload["match_count"])
            for c in payload["citations"]:
                print("   -", c)
            print("\ncontext the agent receives:")
            print(payload["context"][:400])


asyncio.run(main())