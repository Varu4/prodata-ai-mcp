import os
from mcp.server.fastmcp import FastMCP
from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance, 
    generate_report,
    clean_dataset,        
    detect_anomalies,      
    compare_datasets, 
    cluster_data,
    correlation_analysis,
    explain_model,
    generate_dashboard,
    suggest_visualizations,
    generate_sql,  
)

# Create MCP server
import os

port = int(os.environ.get("PORT", 8080))
is_railway = os.environ.get("RAILWAY_ENVIRONMENT") is not None

if is_railway:
    mcp = FastMCP(
        "ProData AI",
        host="0.0.0.0",
        port=port,
    )
else:
    # MCPize manages host/port itself
    mcp = FastMCP("ProData AI")


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


@mcp.tool()
async def clean_dataset_tool(
    csv_data: str,
    drop_duplicates: bool = True,
    fill_numeric: str = "median",
    fill_categorical: str = "mode",
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0,
    strip_whitespace: bool = True,
):
    """Clean a CSV dataset — handles missing values, duplicates, whitespace, and outliers."""
    return await clean_dataset(
        csv_data, drop_duplicates, fill_numeric,
        fill_categorical, remove_outliers,
        outlier_threshold, strip_whitespace
    )

@mcp.tool()
async def detect_anomalies_tool(
    csv_data: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    zscore_threshold: float = 3.0,
    columns: str = "",
):
    """Detect anomalies and outliers in a CSV dataset using Isolation Forest, Z-score, or IQR."""
    return await detect_anomalies(
        csv_data, method, contamination,
        zscore_threshold, columns
    )

@mcp.tool()
async def compare_datasets_tool(
    csv_data_1: str,
    csv_data_2: str,
    label_1: str = "Dataset A",
    label_2: str = "Dataset B",
    match_column: str = "",
):
    """Compare two CSV datasets — schema diff, statistical shifts, distribution changes."""
    return await compare_datasets(
        csv_data_1, csv_data_2,
        label_1, label_2, match_column
    )

@mcp.tool()
async def cluster_data_tool(
    csv_data: str,
    n_clusters: int = 3,
    columns: str = "",
    method: str = "kmeans",
    scale_features: bool = True,
):
    """Cluster/segment rows using K-Means. Returns cluster profiles and distinguishing features."""
    return await cluster_data(csv_data, n_clusters, columns, method, scale_features)

@mcp.tool()
async def correlation_analysis_tool(
    csv_data: str,
    target_column: str = "",
    method: str = "pearson",
    top_n: int = 10,
    threshold: float = 0.0,
):
    """Compute correlation matrix, top pairs, p-values, and multicollinearity warnings."""
    return await correlation_analysis(csv_data, target_column, method, top_n, threshold)

@mcp.tool()
async def explain_model_tool(
    csv_data: str,
    target_column: str,
    audience: str = "business",
    include_recommendations: bool = True,
):
    """Claude-powered plain-English explanation of ML results with actionable recommendations."""
    return await explain_model(csv_data, target_column, audience, include_recommendations)

@mcp.tool()
async def generate_dashboard_tool(
    csv_data: str,
    title: str = "Data Dashboard",
    target_column: str = "",
    theme: str = "dark",
):
    """Generate a self-contained interactive HTML dashboard from a CSV dataset."""
    return await generate_dashboard(csv_data, title, target_column, theme)

@mcp.tool()
async def suggest_visualizations_tool(
    csv_data: str,
    target_column: str = "",
    max_suggestions: int = 8,
):
    """Suggest the best chart types for your dataset based on column types."""
    return await suggest_visualizations(csv_data, target_column, max_suggestions)

@mcp.tool()
async def generate_sql_tool(
    csv_data: str,
    question: str,
    table_name: str = "dataset",
    dialect: str = "standard",
):
    """Claude-powered natural language to SQL — describe what you want, get a query back."""
    return await generate_sql(csv_data, question, table_name, dialect)
    
# =========================
# RUN
# =========================

if __name__ == "__main__":
    import uvicorn
    from starlette.middleware.trustedhost import TrustedHostMiddleware
    port = int(os.environ.get("PORT", 8080))
    app = mcp.streamable_http_app()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    uvicorn.run(app, host="0.0.0.0", port=port)
