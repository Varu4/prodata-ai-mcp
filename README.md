# ProData AI MCP Server

Professional data analysis tool integrated with Claude's Model Context Protocol.

## Features

- **AutoML Training**: Train 6 different models automatically and get the best performer
- **Time Series Forecasting**: Predict future values using Prophet
- **Dataset Analysis**: Comprehensive data quality and statistical analysis
- **Feature Importance**: Identify key drivers of your target variable
- **Report Generation**: Generate professional analysis reports with recommendations

## Quick Start

### Installation

```bash
pip install -r mcp_requirements.txt
```

### Run Server

```bash
python server_http.py
```

Server will be ready at `http://localhost:8080

## Available Tools

1. **train_automl_models** - Train multiple ML models on your data
2. **forecast_timeseries** - Forecast future values with confidence intervals
3. **analyze_dataset** - Get data quality scores and statistics
4. **get_feature_importance** - Identify top drivers of your KPI
5. **generate_report** - Generate comprehensive analysis reports

## Usage with Claude

1. Deploy to Railway/Replit
2. Add to Claude MCP settings
3. Use natural language prompts:
   - "Forecast next 30 days of sales"
   - "What drives our KPI?"
   - "Generate analysis report"

## Technologies

- Python 3.11+
- Pandas, NumPy
- Scikit-learn
- Prophet (Facebook's forecasting library)
- Model Context Protocol

## Author

Varun Walekar

## License

MIT
