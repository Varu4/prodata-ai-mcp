# 🤖 ProData AI — MCP Server for Data Analysis & AutoML

<div align="center">

![ProData AI](https://img.shields.io/badge/ProData_AI-MCP_Server-6C63FF?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=)
![Railway](https://img.shields.io/badge/Deployed_on-Railway-0B0D0E?style=for-the-badge&logo=railway)
![MCPize](https://img.shields.io/badge/Listed_on-MCPize-FF6B35?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Professional data analysis & AutoML — directly inside Claude, Cursor, VS Code, and Windsurf.**

[🚀 Install via MCPize](https://mcpize.com) · [🎮 Try Live Demo](https://varu4-prodata-ai-app-d2bocc.streamlit.app) · [🛒 Get on Gumroad](https://varunanalyze.gumroad.com/l/dgluuk) · [📺 Watch Demo](https://www.youtube.com/watch?v=RLVS7EOylAo)

</div>

---

## ✨ What is ProData AI?

ProData AI is a production-ready **Model Context Protocol (MCP) server** that brings professional-grade data analysis, AutoML training, time series forecasting, and full report generation directly into your AI workflow — no code required.

Just upload a CSV and ask Claude (or any MCP-compatible client) to analyze it. ProData AI handles the rest.

> 🏆 **Gradient Boosting achieved R² = 0.9866** on a retail sales dataset, with marketing spend identified as the #1 revenue driver — all via natural language.

---

## 🛠️ Tools (5 MCP Tools)

| Tool | Description |
|------|-------------|
| `analyze_dataset_tool` | Full statistical analysis — mean, median, std, missing values, correlations, data quality |
| `get_feature_importance_tool` | Identify top features driving a target variable using ML |
| `train_automl_models_tool` | Auto-train & compare multiple ML models, returns best performer with R²/accuracy |
| `forecast_timeseries_tool` | Forecast future values using advanced ML models |
| `generate_report_tool` | Full analysis report including stats, ML insights, and recommendations |

All tools accept raw CSV data via a `csv_data` parameter — no file uploads needed.

---

## ⚡ Quick Install

### Option 1 — MCPize (Recommended, One-Click)

[![Install on Claude Desktop](https://img.shields.io/badge/Install-Claude_Desktop-7C3AED?style=for-the-badge)](https://mcpize.com)
[![Install on Cursor](https://img.shields.io/badge/Install-Cursor-000000?style=for-the-badge)](https://mcpize.com)
[![Install on VS Code](https://img.shields.io/badge/Install-VS_Code-007ACC?style=for-the-badge)](https://mcpize.com)

1. Go to [mcpize.com](https://mcpize.com) and find **ProData AI**
2. Click **Install** for your preferred client
3. Sign in with your MCPize account (free tier: 50 requests/month)
4. Start analyzing data!

### Option 2 — Claude Code CLI

```bash
claude mcp add --transport http prodata-ai https://prodata-ai.mcpize.run \
  --header "API_KEY: your_mcpize_api_key"
```

### Option 3 — Claude Desktop `claude_desktop_config.json`

```json
{
  "mcpServers": {
    "prodata-ai": {
      "type": "http",
      "url": "https://prodata-ai.mcpize.run",
      "headers": {
        "API_KEY": "your_mcpize_api_key"
      }
    }
  }
}
```

> Get your API key from [mcpize.com](https://mcpize.com) after signing up.

---

## 💬 Example Usage

Once connected, just talk to Claude naturally:

```
"Analyze this CSV and tell me about data quality"
"Which features most influence sales revenue?"
"Train ML models on this dataset, target column is revenue"
"Forecast the next 30 days of sales"
"Generate a full report including ML insights"
```

---

## 🏗️ Architecture

```
User (Claude Desktop / Cursor / VS Code)
        ↓  MCP Protocol (HTTP)
MCPize Gateway (auth + billing + routing)
        ↓
Railway Server — server_http.py (FastMCP + Starlette)
        ↓
5 ProData AI Tools (scikit-learn, pandas, statsmodels)
```

---

## 🚀 Self-Hosting / Local Development

### Prerequisites
- Python 3.13+
- pip

### Setup

```bash
git clone https://github.com/Varu4/prodata-ai-mcp.git
cd prodata-ai-mcp
pip install -r requirements.txt
```

### Run locally

```bash
python server_http.py
```

Server starts at `http://localhost:8080/mcp`

### Environment Variables

```bash
# .env (copy from .env.example)
PORT=8080
```

> ⚠️ Never commit real secrets. All sensitive keys go in Railway's dashboard or your local `.env` file (gitignored).

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| MCP Framework | FastMCP + Starlette |
| Runtime | Python 3.13 |
| ML | scikit-learn, pandas, numpy |
| Forecasting | statsmodels |
| Hosting | Railway |
| Marketplace | MCPize (80% revenue share) |
| CI/CD | GitHub Actions → Railway auto-deploy |

---

## 📊 Performance Benchmarks

Tested on a 120-row retail sales dataset:

| Model | R² Score |
|-------|----------|
| **Gradient Boosting** | **0.9866** ✅ Best |
| Random Forest | 0.9741 |
| Linear Regression | 0.8923 |

**Top Feature:** Marketing Spend → Revenue (identified by `get_feature_importance_tool`)

---

## 🗺️ Roadmap

- [x] `analyze_dataset_tool`
- [x] `get_feature_importance_tool`
- [x] `train_automl_models_tool`
- [x] `forecast_timeseries_tool`
- [x] `generate_report_tool`
- [ ] `clean_dataset_tool` *(coming soon)*
- [ ] `generate_dashboard_tool` — returns self-contained HTML dashboard *(coming soon)*

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit PRs.

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👨‍💻 Author

**Varun Walekar** — Data Analyst & AI Developer, Bengaluru

[![GitHub](https://img.shields.io/badge/GitHub-Varu4-181717?style=flat&logo=github)](https://github.com/Varu4)
[![Gumroad](https://img.shields.io/badge/Gumroad-Shop-FF90E8?style=flat)](https://varunanalyze.gumroad.com/l/dgluuk)
[![Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B?style=flat&logo=streamlit)](https://varu4-prodata-ai-app-d2bocc.streamlit.app)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**If ProData AI saved you time, give it a ⭐ — it helps others find it!**

</div>
