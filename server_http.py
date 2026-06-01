from mcp.server.fastmcp import FastMCP
from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance,
    generate_report
)
from fastapi import FastAPI

mcp = FastMCP("ProData AI")

# Tool 1
@mcp.tool()
async def analyze_dataset_tool(
    csv_data: str,
    analysis_type: str = "full"
):
    return await analyze_dataset(csv_data, analysis_type)

# Tool 2
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

# Tool 3
@mcp.tool()
async def forecast_timeseries_tool(
    csv_data: str,
    date_column: str,
    value_column: str,
    periods: int = 30,
    interval_width: float = 0.95
):
    return await forecast_timeseries(
        csv_data,
        date_column,
        value_column,
        periods,
        interval_width
    )

# Tool 4
@mcp.tool()
async def get_feature_importance_tool(
    csv_data: str,
    target_column: str,
    top_n: int = 10,
    feature_columns: list[str] = None
):
    return await get_feature_importance(
        csv_data,
        target_column,
        top_n,
        feature_columns
    )

# Tool 5
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

# Create HTTP MCP app
app = mcp.streamable_http_app()

# Health check
@app.get("/")
async def health():
    return {
        "status": "ok",
        "service": "ProData AI MCP"
    }
