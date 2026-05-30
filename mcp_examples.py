#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# PRODATA AI — MCP INTEGRATION EXAMPLES
# Test all tools locally before deploying
# ════════════════════════════════════════════════════════════════════════════════

import asyncio
import json
from mcp_tools import (
    train_automl_models,
    forecast_timeseries,
    analyze_dataset,
    get_feature_importance,
    generate_report
)

# ════════════════════════════════════════════════════════════════════════════════
# SAMPLE DATA
# ════════════════════════════════════════════════════════════════════════════════

# Sample dataset: E-commerce sales data
SAMPLE_CSV = """date,sales,marketing_spend,traffic,customers,conversion_rate
2024-01-01,1000,500,5000,50,0.01
2024-01-02,1100,550,5500,55,0.01
2024-01-03,1200,600,6000,60,0.01
2024-01-04,950,450,4500,45,0.01
2024-01-05,1300,650,6500,65,0.01
2024-01-06,1400,700,7000,70,0.01
2024-01-07,1150,575,5750,57,0.01
2024-01-08,1500,750,7500,75,0.01
2024-01-09,1600,800,8000,80,0.01
2024-01-10,1700,850,8500,85,0.01
2024-01-11,1550,775,7750,77,0.01
2024-01-12,1800,900,9000,90,0.01
2024-01-13,1900,950,9500,95,0.01
2024-01-14,2000,1000,10000,100,0.01
2024-01-15,1850,925,9250,92,0.01
2024-01-16,2100,1050,10500,105,0.01
2024-01-17,2200,1100,11000,110,0.01
2024-01-18,2300,1150,11500,115,0.01
2024-01-19,2150,1075,10750,107,0.01
2024-01-20,2400,1200,12000,120,0.01
2024-01-21,2500,1250,12500,125,0.01
2024-01-22,2600,1300,13000,130,0.01
2024-01-23,2450,1225,12250,122,0.01
2024-01-24,2700,1350,13500,135,0.01
2024-01-25,2800,1400,14000,140,0.01"""

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 1: ANALYZE DATASET
# ════════════════════════════════════════════════════════════════════════════════

async def example_analyze():
    """Analyze the dataset structure and quality"""
    print("\n" + "="*80)
    print("EXAMPLE 1: ANALYZE DATASET")
    print("="*80)
    
    result = await analyze_dataset(SAMPLE_CSV, analysis_type="full")
    
    print(json.dumps(result, indent=2))
    
    print(f"\n✅ Dataset has {result['dataset_shape']['rows']} rows")
    print(f"✅ Data completeness: {result['data_quality']['completeness']:.1f}%")
    print(f"✅ Missing values: {result['data_quality']['missing_percent']:.1f}%")

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 2: TRAIN AUTOML MODELS
# ════════════════════════════════════════════════════════════════════════════════

async def example_automl():
    """Train multiple models to predict sales"""
    print("\n" + "="*80)
    print("EXAMPLE 2: TRAIN AUTOML MODELS")
    print("="*80)
    
    result = await train_automl_models(
        csv_data=SAMPLE_CSV,
        target_column="sales",
        feature_columns=["marketing_spend", "traffic", "customers"],
        test_size=0.2
    )
    
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print(f"\n✅ Tested {result['models_tested']} models")
        print(f"✅ Winner: {result['best_model']}")
        print(f"✅ Score: {result['best_score']:.4f}")
        print(f"✅ Top features: {', '.join(list(result['top_features'].keys())[:3])}")

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 3: FORECAST TIMESERIES
# ════════════════════════════════════════════════════════════════════════════════

async def example_forecast():
    """Forecast future sales"""
    print("\n" + "="*80)
    print("EXAMPLE 3: FORECAST TIMESERIES")
    print("="*80)
    
    result = await forecast_timeseries(
        csv_data=SAMPLE_CSV,
        date_column="date",
        value_column="sales",
        periods=7,
        interval_width=0.95
    )
    
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print(f"\n✅ Forecasted {result['periods']} periods ahead")
        print(f"✅ Latest value: ${result['latest_value']:.2f}")
        print(f"✅ Projected value: ${result['projected_value']:.2f}")
        print(f"✅ Expected change: {result['change_percent']:.1f}%")
        if result.get('mape'):
            print(f"✅ Model accuracy (MAPE): {result['mape']:.2%}")

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 4: GET FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════════

async def example_drivers():
    """Identify top business drivers"""
    print("\n" + "="*80)
    print("EXAMPLE 4: GET FEATURE IMPORTANCE")
    print("="*80)
    
    result = await get_feature_importance(
        csv_data=SAMPLE_CSV,
        target_column="sales",
        top_n=5
    )
    
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print(f"\n✅ Top driver: {result['top_driver']}")
        print(f"✅ Analyzed {result['total_features']} features")
        print(f"✅ Top features driving sales:")
        for feat, importance in result['top_features'].items():
            print(f"   - {feat}: {importance:.4f}")

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 5: GENERATE REPORT
# ════════════════════════════════════════════════════════════════════════════════

async def example_report():
    """Generate comprehensive analysis report"""
    print("\n" + "="*80)
    print("EXAMPLE 5: GENERATE REPORT")
    print("="*80)
    
    result = await generate_report(
        csv_data=SAMPLE_CSV,
        report_type="detailed",
        include_ml=True,
        target_column="sales"
    )
    
    print(json.dumps(result, indent=2))
    
    if result.get("success"):
        print(f"\n✅ Report Type: {result['report_type']}")
        summary = result['executive_summary']
        print(f"✅ Dataset: {summary['dataset_size']}")
        print(f"✅ Numeric Columns: {summary['numeric_columns']}")
        quality = result['data_quality']
        print(f"✅ Data Quality: {quality['assessment']} ({quality['quality_score']:.1f}%)")
        print(f"✅ Recommendations:")
        for rec in result['recommendations'][:3]:
            print(f"   - {rec}")

# ════════════════════════════════════════════════════════════════════════════════
# EXAMPLE 6: SIMULATED CLAUDE INTERACTION
# ════════════════════════════════════════════════════════════════════════════════

async def example_claude_interaction():
    """Simulate how Claude would interact with the tools"""
    print("\n" + "="*80)
    print("EXAMPLE 6: SIMULATED CLAUDE INTERACTION")
    print("="*80)
    
    print("\nClaude receives prompt:")
    print("─" * 80)
    prompt = "I have e-commerce sales data. Can you forecast the next 7 days and identify what drives sales?"
    print(f"User: {prompt}")
    print("─" * 80)
    
    print("\nClaude's reasoning:")
    print("  1. User wants forecast → call forecast_timeseries")
    print("  2. User wants drivers → call get_feature_importance")
    print("  3. Return insights and actionable recommendations")
    
    print("\nCalling tools in sequence...\n")
    
    # Tool 1: Forecast
    forecast_result = await forecast_timeseries(
        csv_data=SAMPLE_CSV,
        date_column="date",
        value_column="sales",
        periods=7
    )
    
    # Tool 2: Drivers
    drivers_result = await get_feature_importance(
        csv_data=SAMPLE_CSV,
        target_column="sales",
        top_n=3
    )
    
    print("\nClaude's Response:")
    print("─" * 80)
    print(f"📊 **Forecast Results:**")
    if forecast_result.get("success"):
        print(f"   Current sales: ${forecast_result['latest_value']:.2f}")
        print(f"   7-day projection: ${forecast_result['projected_value']:.2f}")
        print(f"   Expected change: {forecast_result['change_percent']:+.1f}%")
    
    print(f"\n🎯 **Key Drivers of Sales:**")
    if drivers_result.get("success"):
        for feat, score in list(drivers_result['top_features'].items())[:3]:
            print(f"   1. {feat} (importance: {score:.1%})")
    
    print(f"\n💡 **Recommendations:**")
    print(f"   • Increase marketing spend to boost traffic")
    print(f"   • Focus on customer acquisition")
    print(f"   • Monitor conversion rate trends")
    print("─" * 80)

# ════════════════════════════════════════════════════════════════════════════════
# RUN ALL EXAMPLES
# ════════════════════════════════════════════════════════════════════════════════

async def run_all_examples():
    """Run all examples"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + "PRODATA AI — MCP TOOL EXAMPLES".center(78) + "║")
    print("║" + "Testing all 5 tools locally".center(78) + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        await example_analyze()
        await example_automl()
        await example_forecast()
        await example_drivers()
        await example_report()
        await example_claude_interaction()
        
        print("\n" + "="*80)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nNext steps:")
        print("  1. Deploy MCP server: python mcp_server.py")
        print("  2. Test in Claude with MCP enabled")
        print("  3. Register with Anthropic MCP registry")
        print("  4. Start earning from tool usage!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_all_examples())
