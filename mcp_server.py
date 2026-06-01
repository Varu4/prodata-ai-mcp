#!/usr/bin/env python3
import json
import logging
import mcp.server.stdio
from mcp.server import Server
from mcp.types import Tool, TextContent  

from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance,
    generate_report
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server = Server("prodata-ai-mcp")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(
            name="train_automl_models",
            description="Train multiple AutoML models on your dataset and get the best performer.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {"type": "string", "description": "CSV data as a string"},
                    "target_column": {"type": "string", "description": "Name of the target column"},
                    "feature_columns": {"type": "array", "items": {"type": "string"}, "description": "List of feature columns"},
                    "test_size": {"type": "number", "description": "Test size (default: 0.2)", "default": 0.2}
                },
                "required": ["csv_data", "target_column"]
            }
        ),
        Tool(
            name="forecast_timeseries",
            description="Forecast future values using Prophet.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {"type": "string", "description": "CSV data"},
                    "date_column": {"type": "string", "description": "Date column name"},
                    "value_column": {"type": "string", "description": "Value column to forecast"},
                    "periods": {"type": "integer", "description": "Periods to forecast", "default": 30},
                    "interval_width": {"type": "number", "description": "Confidence interval", "default": 0.95}
                },
                "required": ["csv_data", "date_column", "value_column"]
            }
        ),
        Tool(
            name="analyze_dataset",
            description="Analyze dataset quality and statistics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {"type": "string", "description": "CSV data"},
                    "analysis_type": {"type": "string", "enum": ["full", "quick", "quality", "statistics"], "default": "full"}
                },
                "required": ["csv_data"]
            }
        ),
        Tool(
            name="get_feature_importance",
            description="Identify key drivers of a target variable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {"type": "string", "description": "CSV data"},
                    "target_column": {"type": "string", "description": "Target variable"},
                    "top_n": {"type": "integer", "description": "Top features", "default": 10},
                    "feature_columns": {"type": "array", "items": {"type": "string"}, "description": "Features to analyze"}
                },
                "required": ["csv_data", "target_column"]
            }
        ),
        Tool(
            name="generate_report",
            description="Generate comprehensive analysis report.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {"type": "string", "description": "CSV data"},
                    "report_type": {"type": "string", "enum": ["executive", "detailed", "technical"], "default": "detailed"},
                    "include_ml": {"type": "boolean", "description": "Include ML analysis", "default": True},
                    "target_column": {"type": "string", "description": "Target for ML"}
                },
                "required": ["csv_data"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Calling tool: {name}")
    try:
        if name == "train_automl_models":
            result = await train_automl_models(
                csv_data=arguments.get("csv_data"),
                target_column=arguments.get("target_column"),
                feature_columns=arguments.get("feature_columns", []),
                test_size=arguments.get("test_size", 0.2)
            )
        elif name == "forecast_timeseries":
            result = await forecast_timeseries(
                csv_data=arguments.get("csv_data"),
                date_column=arguments.get("date_column"),
                value_column=arguments.get("value_column"),
                periods=arguments.get("periods", 30),
                interval_width=arguments.get("interval_width", 0.95)
            )
        elif name == "analyze_dataset":
            result = await analyze_dataset(
                csv_data=arguments.get("csv_data"),
                analysis_type=arguments.get("analysis_type", "full")
            )
        elif name == "get_feature_importance":
            result = await get_feature_importance(
                csv_data=arguments.get("csv_data"),
                target_column=arguments.get("target_column"),
                top_n=arguments.get("top_n", 10),
                feature_columns=arguments.get("feature_columns", [])
            )
        elif name == "generate_report":
            result = await generate_report(
                csv_data=arguments.get("csv_data"),
                report_type=arguments.get("report_type", "detailed"),
                include_ml=arguments.get("include_ml", True),
                target_column=arguments.get("target_column", None)
            )
        else:
            result = {"error": f"Unknown tool: {name}"}
        return [TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        logger.error(error_msg)
        return [TextContent(type="text", text=json.dumps({"error": error_msg}))]

async def main():
    logger.info("Starting ProData AI MCP Server...")
    async with mcp.server.stdio.stdio_server(server) as (read_stream, write_stream):
        logger.info("Server ready and listening...")
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
