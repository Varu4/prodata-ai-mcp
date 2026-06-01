from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ProData AI")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello {name}"

app = mcp.streamable_http_app()
