import os
from mcp.server.fastmcp import FastMCP
from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance,
    generate_report
)

# Create MCP server
port = int(os.environ.get("PORT", 8080))
mcp = FastMCP(
    "ProData AI",
    host="0.0.0.0",
    port=port
    cors_origins=["*"],
    allowed_hosts=["*"],
)


# =========================
# MCP TOOLS
# =========================

@mcp.tool()
async def analyze_dataset_tool(
    csv_data: str,
    analysis_type: str = "full"
):
    """Analyze a CSV dataset and return statistics, correlations, and insights."""
    return await analyze_dataset(csv_data, analysis_type)


@mcp.tool()
async def train_automl_models_tool(
    csv_data: str,
    target_column: str,
    feature_columns: list[str] = None,
    test_size: float = 0.2
):
    """Train multiple ML models automatically and return the best performing one."""
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
    """Forecast future values of a time series using advanced ML models."""
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
    """Return the most important features driving a target variable."""
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
    """Generate a full analysis report including stats, ML models, and insights."""
    return await generate_report(
        csv_data,
        report_type,
        include_ml,
        target_column
    )


# =========================
# RUN
# =========================

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
