#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════════════════════════
# PRODATA AI — MCP TOOLS IMPLEMENTATION
# Core functions exposed through MCP
# ════════════════════════════════════════════════════════════════════════════════

import io
import json
import logging
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from prophet import Prophet

logger = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════════

def load_csv_data(csv_data: str) -> pd.DataFrame:
    """Load CSV data from string or file path"""
    try:
        # Try to treat as CSV string first
        if '\n' in csv_data or ',' in csv_data:
            return pd.read_csv(io.StringIO(csv_data))
        else:
            # Try to load as file path
            return pd.read_csv(csv_data)
    except Exception as e:
        raise ValueError(f"Error loading CSV data: {str(e)}")

def get_numeric_categorical_cols(df: pd.DataFrame) -> tuple:
    """Get numeric and categorical columns"""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical

# ════════════════════════════════════════════════════════════════════════════════
# TOOL 1: TRAIN AUTOML MODELS
# ════════════════════════════════════════════════════════════════════════════════

async def train_automl_models(
    csv_data: str,
    target_column: str,
    feature_columns: List[str] = None,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """
    Train 6 different models and return the best one with metrics
    """
    try:
        df = load_csv_data(csv_data)
        
        # Validate target column
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found in dataset"}
        
        # Get features
        if not feature_columns or len(feature_columns) == 0:
            numeric, _ = get_numeric_categorical_cols(df)
            feature_columns = [c for c in numeric if c != target_column]
        
        if not feature_columns:
            return {"error": "No numeric features found for training"}
        
        # Prepare data
        X = pd.get_dummies(df[feature_columns], drop_first=True).fillna(0)
        y = df[target_column].fillna(df[target_column].median())
        
        # Determine if classification or regression
        is_classification = (y.nunique() < 15) and (y.nunique() > 1)
        
        # Encode labels for classification
        if is_classification:
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
        else:
            y_encoded = y.values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42
        )
        
        # Define models
        if is_classification:
            models = [
                ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
                ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
                ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
                ("Decision Tree", DecisionTreeClassifier(max_depth=10, random_state=42)),
            ]
        else:
            models = [
                ("Random Forest", RandomForestRegressor(n_estimators=100, random_state=42)),
                ("Gradient Boosting", GradientBoostingRegressor(random_state=42)),
                ("Linear Regression", LinearRegression()),
                ("Ridge Regression", Ridge()),
                ("Lasso Regression", Lasso(max_iter=5000)),
                ("Decision Tree", DecisionTreeRegressor(max_depth=10, random_state=42)),
            ]
        
        # Train models
        results = []
        best_score = -999
        best_model_name = None
        best_model = None
        
        for name, model in models:
            try:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                
                if is_classification:
                    score = accuracy_score(y_test, predictions)
                    metric = "accuracy"
                else:
                    score = r2_score(y_test, predictions)
                    metric = "r2"
                
                results.append({
                    "model": name,
                    "score": float(score),
                    "metric": metric,
                    "status": "success"
                })
                
                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model
            except Exception as e:
                results.append({
                    "model": name,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Get feature importance
        importances = {}
        if hasattr(best_model, 'feature_importances_'):
            imp = pd.Series(best_model.feature_importances_, index=X.columns)
            imp_sorted = imp.sort_values(ascending=False)
            importances = imp_sorted.head(10).to_dict()
        
        return {
            "success": True,
            "task": "automl",
            "target": target_column,
            "type": "classification" if is_classification else "regression",
            "models_tested": len(results),
            "best_model": best_model_name,
            "best_score": float(best_score),
            "all_results": results,
            "top_features": importances,
            "training_samples": len(X_train),
            "test_samples": len(X_test)
        }
    
    except Exception as e:
        logger.error(f"Error in train_automl_models: {str(e)}")
        return {"error": f"AutoML training failed: {str(e)}"}

# ════════════════════════════════════════════════════════════════════════════════
# TOOL 2: FORECAST TIMESERIES
# ════════════════════════════════════════════════════════════════════════════════

async def forecast_timeseries(
    csv_data: str,
    date_column: str,
    value_column: str,
    periods: int = 30,
    interval_width: float = 0.95
) -> Dict[str, Any]:
    """
    Forecast future values using Prophet
    """
    try:
        df = load_csv_data(csv_data)
        
        # Validate columns
        if date_column not in df.columns:
            return {"error": f"Date column '{date_column}' not found"}
        if value_column not in df.columns:
            return {"error": f"Value column '{value_column}' not found"}
        
        # Prepare data
        df_ts = df[[date_column, value_column]].dropna().sort_values(date_column).copy()
        df_ts.columns = ['ds', 'y']
        
        # Convert date column
        df_ts['ds'] = pd.to_datetime(df_ts['ds'])
        
        if len(df_ts) < 20:
            return {"error": f"Need at least 20 data points for forecasting (found {len(df_ts)})"}
        
        # Train Prophet
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            interval_width=interval_width
        )
        
        # Split data for validation
        train_size = int(len(df_ts) * 0.8)
        train_df = df_ts.iloc[:train_size]
        test_df = df_ts.iloc[train_size:]
        
        # Fit model
        model.fit(train_df)
        
        # Generate forecast
        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)
        
        # Validation metrics
        test_forecast = model.predict(test_df[['ds']])
        mape = np.mean(np.abs((test_df['y'].values - test_forecast['yhat'].values) / test_df['y'].values)) if len(test_df) > 0 else None
        
        # Get latest and projected values
        latest_value = float(df_ts['y'].iloc[-1])
        projected_value = float(forecast['yhat'].iloc[-1])
        
        # Get forecast data
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records')
        
        return {
            "success": True,
            "task": "forecast",
            "value_column": value_column,
            "periods": periods,
            "historical_points": len(df_ts),
            "latest_value": latest_value,
            "projected_value": projected_value,
            "change_percent": float((projected_value - latest_value) / latest_value * 100) if latest_value != 0 else 0,
            "mape": float(mape) if mape else None,
            "confidence_interval": interval_width,
            "forecast_dates": [str(d['ds']) for d in forecast_data],
            "forecast_values": [float(d['yhat']) for d in forecast_data],
            "forecast_lower": [float(d['yhat_lower']) for d in forecast_data],
            "forecast_upper": [float(d['yhat_upper']) for d in forecast_data]
        }
    
    except Exception as e:
        logger.error(f"Error in forecast_timeseries: {str(e)}")
        return {"error": f"Forecasting failed: {str(e)}"}

# ════════════════════════════════════════════════════════════════════════════════
# TOOL 3: ANALYZE DATASET
# ════════════════════════════════════════════════════════════════════════════════

async def analyze_dataset(
    csv_data: str,
    analysis_type: str = "full"
) -> Dict[str, Any]:
    """
    Comprehensive dataset analysis
    """
    try:
        df = load_csv_data(csv_data)
        numeric, categorical = get_numeric_categorical_cols(df)
        
        analysis = {
            "success": True,
            "task": "analyze",
            "dataset_shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "numeric_columns": len(numeric),
            "categorical_columns": len(categorical),
        }
        
        if analysis_type in ["full", "quality"]:
            # Data quality
            missing_total = int(df.isna().sum().sum())
            missing_percent = float(missing_total / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] * df.shape[1] > 0 else 0
            duplicates = int(df.duplicated().sum())
            
            analysis.update({
                "data_quality": {
                    "missing_values": missing_total,
                    "missing_percent": missing_percent,
                    "duplicate_rows": duplicates,
                    "duplicate_percent": float(duplicates / df.shape[0] * 100) if df.shape[0] > 0 else 0,
                    "completeness": float((1 - missing_percent/100) * 100)
                }
            })
        
        if analysis_type in ["full", "statistics"]:
            # Statistics
            stats = {}
            for col in numeric[:10]:  # Top 10 numeric columns
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    stats[col] = {
                        "mean": float(col_data.mean()),
                        "median": float(col_data.median()),
                        "std": float(col_data.std()),
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "count": int(len(col_data))
                    }
            
            analysis["statistics"] = stats
        
        if analysis_type in ["full", "quality"]:
            # Missing values by column
            missing_by_col = {}
            for col in df.columns:
                missing = int(df[col].isna().sum())
                if missing > 0:
                    missing_by_col[col] = {
                        "count": missing,
                        "percent": float(missing / len(df) * 100)
                    }
            
            analysis["missing_by_column"] = missing_by_col
        
        analysis["columns"] = list(df.columns)
        analysis["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error in analyze_dataset: {str(e)}")
        return {"error": f"Dataset analysis failed: {str(e)}"}

# ════════════════════════════════════════════════════════════════════════════════
# TOOL 4: GET FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════════

async def get_feature_importance(
    csv_data: str,
    target_column: str,
    top_n: int = 10,
    feature_columns: List[str] = None
) -> Dict[str, Any]:
    """
    Analyze feature importance for predicting target variable
    """
    try:
        df = load_csv_data(csv_data)
        
        if target_column not in df.columns:
            return {"error": f"Target column '{target_column}' not found"}
        
        # Get features
        if not feature_columns or len(feature_columns) == 0:
            numeric, _ = get_numeric_categorical_cols(df)
            feature_columns = [c for c in numeric if c != target_column]
        
        if not feature_columns:
            return {"error": "No numeric features found"}
        
        # Prepare data
        X = df[feature_columns].fillna(0)
        y = df[target_column].fillna(df[target_column].median())
        
        # Determine task type
        is_classification = (y.nunique() < 15) and (y.nunique() > 1)
        
        # Train model
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X, y)
        
        # Get importance
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance_sorted = importance.sort_values(ascending=False)
        
        # Top N features
        top_features = importance_sorted.head(top_n).to_dict()
        
        return {
            "success": True,
            "task": "feature_importance",
            "target": target_column,
            "type": "classification" if is_classification else "regression",
            "total_features": len(feature_columns),
            "top_features": {k: float(v) for k, v in top_features.items()},
            "top_driver": list(top_features.keys())[0] if top_features else None,
            "importance_percent": {k: float(v * 100) for k, v in top_features.items()},
            "cumulative_importance": float(sum(top_features.values())),
            "normalized": True
        }
    
    except Exception as e:
        logger.error(f"Error in get_feature_importance: {str(e)}")
        return {"error": f"Feature importance analysis failed: {str(e)}"}

# ════════════════════════════════════════════════════════════════════════════════
# TOOL 5: GENERATE REPORT
# ════════════════════════════════════════════════════════════════════════════════

async def generate_report(
    csv_data: str,
    report_type: str = "detailed",
    include_ml: bool = True,
    target_column: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate comprehensive analysis report
    """
    try:
        df = load_csv_data(csv_data)
        numeric, categorical = get_numeric_categorical_cols(df)
        
        report = {
            "success": True,
            "task": "generate_report",
            "report_type": report_type,
            "generated_at": str(pd.Timestamp.now()),
        }
        
        # Executive Summary
        summary = {
            "dataset_size": f"{df.shape[0]:,} rows × {df.shape[1]} columns",
            "numeric_columns": len(numeric),
            "categorical_columns": len(categorical),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
        }
        report["executive_summary"] = summary
        
        # Data Quality Assessment
        missing_percent = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if df.shape[0] * df.shape[1] > 0 else 0
        data_quality = "Excellent" if missing_percent < 5 else "Good" if missing_percent < 15 else "Fair" if missing_percent < 30 else "Poor"
        
        report["data_quality"] = {
            "assessment": data_quality,
            "missing_percent": missing_percent,
            "quality_score": float(100 - missing_percent)
        }
        
        # Column Analysis
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()),
                "null_count": int(df[col].isna().sum()),
                "unique_values": int(df[col].nunique())
            }
            if col in numeric:
                col_data = df[col].dropna()
                col_info["stats"] = {
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max())
                }
            columns_info.append(col_info)
        
        report["columns"] = columns_info
        
        # ML Analysis if requested
        if include_ml and target_column and target_column in df.columns:
            ml_result = await train_automl_models(csv_data, target_column)
            if "error" not in ml_result:
                report["ml_analysis"] = ml_result
        
        # Recommendations
        recommendations = []
        if missing_percent > 10:
            recommendations.append(f"High missing data ({missing_percent:.1f}%). Consider data imputation or removal.")
        if int(df.duplicated().sum()) > 0:
            recommendations.append(f"Found {int(df.duplicated().sum())} duplicate rows. Remove before analysis.")
        if len(numeric) < 2:
            recommendations.append("Limited numeric columns. Consider feature engineering.")
        
        if not recommendations:
            recommendations.append("Dataset quality looks good. Ready for analysis.")
        
        report["recommendations"] = recommendations
        
        return report
    
    except Exception as e:
        logger.error(f"Error in generate_report: {str(e)}")
        return {"error": f"Report generation failed: {str(e)}"}
