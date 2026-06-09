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
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score
from prophet import Prophet
from sklearn.ensemble import IsolationForest

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

 
 
def _parse_csv(csv_data: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(csv_data.strip()))
 
 
# ══════════════════════════════════════════════════════════════
# clean_dataset
# ══════════════════════════════════════════════════════════════
async def clean_dataset(
    csv_data: str,
    drop_duplicates: bool = True,
    fill_numeric: str = "median",
    fill_categorical: str = "mode",
    remove_outliers: bool = False,
    outlier_threshold: float = 3.0,
    strip_whitespace: bool = True,
) -> dict:
    try:
        df = _parse_csv(csv_data)
        original_shape = df.shape
        report = []
 
        # 1. Strip whitespace
        if strip_whitespace:
            str_cols = df.select_dtypes(include="object").columns
            for col in str_cols:
                df[col] = df[col].str.strip()
            if len(str_cols):
                report.append(f"Stripped whitespace from {len(str_cols)} text columns: {list(str_cols)}")
 
        # 2. Drop duplicates
        if drop_duplicates:
            before = len(df)
            df = df.drop_duplicates()
            removed = before - len(df)
            report.append(f"Removed {removed} duplicate rows" if removed else "No duplicate rows found")
 
        # 3. Handle missing values
        missing_before = df.isnull().sum()
        total_missing = missing_before.sum()
 
        if total_missing > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(include="object").columns
 
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if fill_numeric == "median":
                        df[col].fillna(df[col].median(), inplace=True)
                    elif fill_numeric == "mean":
                        df[col].fillna(df[col].mean(), inplace=True)
                    elif fill_numeric == "zero":
                        df[col].fillna(0, inplace=True)
                    elif fill_numeric == "drop":
                        df = df.dropna(subset=[col])
 
            for col in cat_cols:
                if df[col].isnull().sum() > 0:
                    if fill_categorical == "mode":
                        mode_val = df[col].mode()
                        df[col].fillna(mode_val[0] if len(mode_val) else "Unknown", inplace=True)
                    elif fill_categorical == "unknown":
                        df[col].fillna("Unknown", inplace=True)
                    elif fill_categorical == "drop":
                        df = df.dropna(subset=[col])
 
            filled_cols = missing_before[missing_before > 0].to_dict()
            report.append(f"Filled {total_missing} missing values: {filled_cols}")
        else:
            report.append("No missing values found")
 
        # 4. Remove outliers
        if remove_outliers:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            before = len(df)
            for col in numeric_cols:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df = df[np.abs((df[col] - mean) / std) <= outlier_threshold]
            removed_outliers = before - len(df)
            report.append(
                f"Removed {removed_outliers} outlier rows (Z-score > {outlier_threshold})"
                if removed_outliers else "No outliers removed"
            )
 
        final_shape = df.shape
        missing_after = df.isnull().sum().sum()
        completeness = round((1 - missing_after / (df.shape[0] * df.shape[1])) * 100, 2) if df.size > 0 else 100.0
 
        return {
            "success": True,
            "task": "clean_dataset",
            "original_shape": {"rows": original_shape[0], "columns": original_shape[1]},
            "cleaned_shape": {"rows": final_shape[0], "columns": final_shape[1]},
            "rows_removed": original_shape[0] - final_shape[0],
            "completeness_after_cleaning": f"{completeness}%",
            "changes_made": report,
            "cleaned_csv": df.to_csv(index=False),
            "column_dtypes": df.dtypes.astype(str).to_dict(),
        }
 
    except Exception as e:
        return {"success": False, "task": "clean_dataset", "error": str(e)}
 
 
# ══════════════════════════════════════════════════════════════
# detect_anomalies
# ══════════════════════════════════════════════════════════════
async def detect_anomalies(
    csv_data: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    zscore_threshold: float = 3.0,
    columns: str = "",
) -> dict:
    try:
        df = _parse_csv(csv_data)
 
        # Select columns
        if columns.strip():
            target_cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        else:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 
        if not target_cols:
            return {
                "success": False,
                "task": "detect_anomalies",
                "error": "No numeric columns found or specified.",
            }
 
        numeric_df = df[target_cols].copy()
        anomaly_mask = pd.Series([False] * len(df), index=df.index)
 
        if method == "isolation_forest":
            clf = IsolationForest(
                contamination=max(0.01, min(0.5, contamination)),
                random_state=42,
                n_estimators=100,
            )
            preds = clf.fit_predict(numeric_df.fillna(numeric_df.median()))
            anomaly_mask = pd.Series(preds == -1, index=df.index)
            scores = clf.decision_function(numeric_df.fillna(numeric_df.median()))
            anomaly_scores = pd.Series(np.round(scores, 4), index=df.index)
 
        elif method == "zscore":
            z_scores = np.abs((numeric_df - numeric_df.mean()) / numeric_df.std())
            anomaly_mask = (z_scores > zscore_threshold).any(axis=1)
            anomaly_scores = z_scores.max(axis=1).round(4)
 
        elif method == "iqr":
            Q1 = numeric_df.quantile(0.25)
            Q3 = numeric_df.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            anomaly_mask = ((numeric_df < lower) | (numeric_df > upper)).any(axis=1)
            deviation = ((numeric_df - numeric_df.median()).abs() / (IQR + 1e-9))
            anomaly_scores = deviation.max(axis=1).round(4)
 
        else:
            return {"success": False, "task": "detect_anomalies", "error": f"Unknown method: {method}"}
 
        anomalous_df = df[anomaly_mask].copy()
        anomalous_df["__anomaly_score__"] = anomaly_scores[anomaly_mask].values
 
        col_stats = {}
        for col in target_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            col_stats[col] = {
                "mean": round(df[col].mean(), 4),
                "std": round(df[col].std(), 4),
                "min": round(df[col].min(), 4),
                "max": round(df[col].max(), 4),
                "outliers_iqr": int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()),
            }
 
        return {
            "success": True,
            "task": "detect_anomalies",
            "method": method,
            "total_rows": len(df),
            "anomalies_found": int(anomaly_mask.sum()),
            "anomaly_percent": round(anomaly_mask.sum() / len(df) * 100, 2),
            "columns_analyzed": target_cols,
            "anomalous_rows": anomalous_df.head(50).to_dict(orient="records"),
            "anomalous_indices": anomaly_mask[anomaly_mask].index.tolist()[:50],
            "column_stats": col_stats,
            "clean_csv": df[~anomaly_mask].to_csv(index=False),
        }
 
    except Exception as e:
        return {"success": False, "task": "detect_anomalies", "error": str(e)}
 
 
# ══════════════════════════════════════════════════════════════
# compare_datasets
# ══════════════════════════════════════════════════════════════
async def compare_datasets(
    csv_data_1: str,
    csv_data_2: str,
    label_1: str = "Dataset A",
    label_2: str = "Dataset B",
    match_column: str = "",
) -> dict:
    try:
        df1 = _parse_csv(csv_data_1)
        df2 = _parse_csv(csv_data_2)
 
        result = {
            "success": True,
            "task": "compare_datasets",
            "labels": [label_1, label_2],
        }
 
        # 1. Shape
        result["shape"] = {
            label_1: {"rows": df1.shape[0], "columns": df1.shape[1]},
            label_2: {"rows": df2.shape[0], "columns": df2.shape[1]},
            "row_diff": df2.shape[0] - df1.shape[0],
            "col_diff": df2.shape[1] - df1.shape[1],
        }
 
        # 2. Schema diff
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        common_cols = cols1 & cols2
        result["schema"] = {
            "columns_only_in_1": sorted(list(cols1 - cols2)),
            "columns_only_in_2": sorted(list(cols2 - cols1)),
            "common_columns": sorted(list(common_cols)),
            "dtype_changes": {},
        }
        for col in common_cols:
            t1 = str(df1[col].dtype)
            t2 = str(df2[col].dtype)
            if t1 != t2:
                result["schema"]["dtype_changes"][col] = {label_1: t1, label_2: t2}
 
        # 3. Statistical comparison
        numeric_common = [
            c for c in common_cols
            if pd.api.types.is_numeric_dtype(df1[c]) and pd.api.types.is_numeric_dtype(df2[c])
        ]
        stats_comparison = {}
        for col in numeric_common:
            s1 = df1[col].dropna()
            s2 = df2[col].dropna()
            mean_change = round(s2.mean() - s1.mean(), 4)
            mean_change_pct = round((mean_change / s1.mean() * 100), 2) if s1.mean() != 0 else None
            stats_comparison[col] = {
                label_1: {
                    "mean": round(s1.mean(), 4),
                    "median": round(s1.median(), 4),
                    "std": round(s1.std(), 4),
                    "min": round(s1.min(), 4),
                    "max": round(s1.max(), 4),
                    "missing": int(df1[col].isnull().sum()),
                },
                label_2: {
                    "mean": round(s2.mean(), 4),
                    "median": round(s2.median(), 4),
                    "std": round(s2.std(), 4),
                    "min": round(s2.min(), 4),
                    "max": round(s2.max(), 4),
                    "missing": int(df2[col].isnull().sum()),
                },
                "mean_change": mean_change,
                "mean_change_pct": f"{mean_change_pct}%" if mean_change_pct is not None else "N/A",
                "distribution_shift": "significant" if abs(mean_change_pct or 0) > 10 else "minor",
            }
        result["statistics"] = stats_comparison
 
        # 4. Categorical comparison
        cat_common = [c for c in common_cols if df1[c].dtype == "object" and df2[c].dtype == "object"]
        cat_comparison = {}
        for col in cat_common:
            vals1 = set(df1[col].dropna().unique())
            vals2 = set(df2[col].dropna().unique())
            cat_comparison[col] = {
                "unique_values_in_1": len(vals1),
                "unique_values_in_2": len(vals2),
                "new_values_in_2": sorted(list(vals2 - vals1))[:20],
                "removed_values_from_1": sorted(list(vals1 - vals2))[:20],
                "top_value_1": str(df1[col].value_counts().index[0]) if len(df1[col].dropna()) else None,
                "top_value_2": str(df2[col].value_counts().index[0]) if len(df2[col].dropna()) else None,
            }
        result["categorical"] = cat_comparison
 
        # 5. Missing value changes
        missing_comparison = {}
        for col in common_cols:
            m1 = int(df1[col].isnull().sum())
            m2 = int(df2[col].isnull().sum())
            if m1 != m2:
                missing_comparison[col] = {label_1: m1, label_2: m2, "change": m2 - m1}
        result["missing_value_changes"] = missing_comparison
 
        # 6. Row-level diff
        if match_column and match_column in common_cols:
            keys1 = set(df1[match_column].astype(str))
            keys2 = set(df2[match_column].astype(str))
            result["row_diff"] = {
                "match_column": match_column,
                "rows_only_in_1": sorted(list(keys1 - keys2))[:30],
                "rows_only_in_2": sorted(list(keys2 - keys1))[:30],
                "common_rows": len(keys1 & keys2),
            }
 
        # 7. Summary
        significant_shifts = [
            col for col, v in stats_comparison.items()
            if v.get("distribution_shift") == "significant"
        ]
        result["summary"] = {
            "overall_similarity": (
                "high" if len(significant_shifts) == 0
                and not result["schema"]["columns_only_in_1"]
                and not result["schema"]["columns_only_in_2"]
                else "moderate" if len(significant_shifts) <= 2 else "low"
            ),
            "significant_shifts_in": significant_shifts,
            "schema_changed": bool(
                result["schema"]["columns_only_in_1"]
                or result["schema"]["columns_only_in_2"]
                or result["schema"]["dtype_changes"]
            ),
            "recommendation": (
                "Datasets look very similar — minor differences only."
                if not significant_shifts
                else f"Notable distribution shifts in: {', '.join(significant_shifts)}. Investigate before merging."
            ),
        }
 
        return result
 
    except Exception as e:
        return {"success": False, "task": "compare_datasets", "error": str(e)}
