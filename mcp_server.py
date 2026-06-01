#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# PRODATA AI — MCP SERVER v1.0
# Model Context Protocol Implementation
# Exposes ProData AI tools to Claude and other AI platforms
# ════════════════════════════════════════════════════════════════════════════════

import json
import logging
from typing import Any
import mcp.server.stdio
from mcp.server import Server
from mcp.types import Tool, TextContent

# Import tool implementations
from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance,
    generate_report
)

# ════════════════════════════════════════════════════════════════════════════════
# SETUP LOGGING
# ════════════════════════════════════════════════════════════════════════════════

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# MCP SERVER SETUP
# ════════════════════════════════════════════════════════════════════════════════

server = Server("prodata-ai-mcp")

# ════════════════════════════════════════════════════════════════════════════════
# TOOL 1: TRAIN AUTOML MODELS
# ════════════════════════════════════════════════════════════════════════════════

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """List all available ProData AI tools"""
    return [
        Tool(
            name="train_automl_models",
            description="Train multiple AutoML models on your dataset and get the best performer. Automatically trains 6 different models (Random Forest, Gradient Boosting, Linear Regression, Ridge, Lasso, Decision Tree) and returns the winner with performance metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {
                        "type": "string",
                        "description": "CSV data as a string or path to CSV file. First row should be headers."
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target column to predict"
                    },
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of feature column names to use. If empty, all numeric columns except target are used."
                    },
                    "test_size": {
                        "type": "number",
                        "description": "Fraction of data to use for testing (0.1 to 0.5). Default: 0.2",
                        "default": 0.2
                    }
                },
                "required": ["csv_data", "target_column"]
            }
        ),
        Tool(
            name="forecast_timeseries",
            description="Forecast future values using time series analysis (Prophet). Predicts future values for your time series data with confidence intervals.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {
                        "type": "string",
                        "description": "CSV data with date and value columns"
                    },
                    "date_column": {
                        "type": "string",
                        "description": "Name of the date column (e.g., 'Date', 'timestamp')"
                    },
                    "value_column": {
                        "type": "string",
                        "description": "Name of the value column to forecast"
                    },
                    "periods": {
                        "type": "integer",
                        "description": "Number of periods to forecast ahead (default: 30 days)",
                        "default": 30
                    },
                    "interval_width": {
                        "type": "number",
                        "description": "Confidence interval width (0.0 to 1.0, default: 0.95)",
                        "default": 0.95
                    }
                },
                "required": ["csv_data", "date_column", "value_column"]
            }
        ),
        Tool(
            name="analyze_dataset",
            description="Comprehensive dataset analysis including data quality checks, statistical summaries, missing value analysis, and data type profiling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {
                        "type": "string",
                        "description": "CSV data to analyze"
                    },
                    "analysis_type": {
                        "type": "string",
                        "enum": ["full", "quick", "quality", "statistics"],
                        "description": "Type of analysis: full (all), quick (summary), quality (data quality), statistics (statistical summary)",
                        "default": "full"
                    }
                },
                "required": ["csv_data"]
            }
        ),
        Tool(
            name="get_feature_importance",
            description="Analyze which features (columns) have the most impact on a target variable. Uses Random Forest feature importance to identify key drivers.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {
                        "type": "string",
                        "description": "CSV data with features and target column"
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Name of the target variable (KPI)"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of top features to return (default: 10)",
                        "default": 10
                    },
                    "feature_columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific feature columns to analyze (optional, uses all numeric by default)"
                    }
                },
                "required": ["csv_data", "target_column"]
            }
        ),
        Tool(
            name="generate_report",
            description="Generate a comprehensive analysis report including all insights, visualizations, and recommendations. Returns a structured report with executive summary, findings, and next steps.",
            inputSchema={
                "type": "object",
                "properties": {
                    "csv_data": {
                        "type": "string",
                        "description": "CSV data to analyze"
                    },
                    "report_type": {
                        "type": "string",
                        "enum": ["executive", "detailed", "technical"],
                        "description": "Type of report: executive (summary), detailed (comprehensive), technical (detailed analysis)",
                        "default": "detailed"
                    },
                    "include_ml": {
                        "type": "boolean",
                        "description": "Include machine learning analysis",
                        "default": True
                    },
                    "target_column": {
                        "type": "string",
                        "description": "Target column for ML analysis (optional)"
                    }
                },
                "required": ["csv_data"]
            }
        )
    ]

# ════════════════════════════════════════════════════════════════════════════════
# TOOL EXECUTION
# ════════════════════════════════════════════════════════════════════════════════

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute ProData AI tools"""
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

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

async def main():
    """Start the MCP server"""
    logger.info("Starting ProData AI MCP Server...")
    logger.info("Available tools: train_automl_models, forecast_timeseries, analyze_dataset, get_feature_importance, generate_report")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server is ready and listening...")
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
