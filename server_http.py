from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("ProData AI")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello {name}"

app = FastAPI()

@app.get("/")
def health():
    return {
        "status": "ok",
        "service": "ProData AI MCP"
    }

app.mount("/mcp", mcp.streamable_http_app())
