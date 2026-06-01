from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, Mount

from mcp.server.fastmcp import FastMCP

from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance,
    generate_report
)

# Create MCP server
mcp = FastMCP("ProData AI")


# =========================
# MCP TOOLS
# =========================

@mcp.tool()
async def analyze_dataset_tool(
    csv_data: str,
    analysis_type: str = "full"
):
    return await analyze_dataset(csv_data, analysis_type)


@mcp.tool()
async def train_automl_models_tool(
    csv_data: str,
    target_column: str,
    feature_columns: list[str] = None,
    test_size: float = 0.2
):
    return await train_automl_models(
        csv_data,
        target_column,
        feature_columns,
        test_size
    )


@mcp.tool()
async def forecast_timeseries_tool(
    csv_data: str,
    date_column: str,
    value_column: str,
    periods: int = 30
):
    return await forecast_timeseries(
        csv_data,
        date_column,
        value_column,
        periods
    )


@mcp.tool()
async def get_feature_importance_tool(
    csv_data: str,
    target_column: str,
    top_n: int = 10
):
    return await get_feature_importance(
        csv_data,
        target_column,
        top_n
    )


@mcp.tool()
async def generate_report_tool(
    csv_data: str,
    report_type: str = "detailed",
    include_ml: bool = True,
    target_column: str = None
):
    return await generate_report(
        csv_data,
        report_type,
        include_ml,
        target_column
    )


# =========================
# HEALTH CHECK
# =========================

async def health(request):
    return JSONResponse({
        "status": "ok",
        "service": "ProData AI MCP"
    })


# MCP HTTP APP
mcp_app = mcp.streamable_http_app()

# Main Railway App
app = Starlette(
    routes=[
        Route("/", health),
        Mount("/mcp", app=mcp_app)
    ]
)
