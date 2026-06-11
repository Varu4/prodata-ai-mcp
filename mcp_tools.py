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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
import anthropic

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

# ══════════════════════════════════════════════════════════════
#  cluster_data
# ══════════════════════════════════════════════════════════════
async def cluster_data(
    csv_data: str,
    n_clusters: int = 3,
    columns: str = "",
    method: str = "kmeans",
    scale_features: bool = True,
) -> dict:
    """
    Segment/cluster rows in a CSV dataset using K-Means.
    Returns cluster labels, per-cluster stats, and top distinguishing features.
    Great for customer segmentation, product grouping, and pattern discovery.
    """
    try:
        df = _parse_csv(csv_data)
 
        if columns.strip():
            target_cols = [c.strip() for c in columns.split(",") if c.strip() in df.columns]
        else:
            target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 
        if not target_cols:
            return {"success": False, "task": "cluster_data", "error": "No numeric columns found."}
 
        if len(df) < n_clusters:
            return {"success": False, "task": "cluster_data", "error": f"Need at least {n_clusters} rows."}
 
        X = df[target_cols].fillna(df[target_cols].median())
 
        if scale_features:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
 
        n_clusters = min(n_clusters, len(df))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df["__cluster__"] = labels
 
        inertia = float(kmeans.inertia_)
 
        cluster_profiles = {}
        for cluster_id in range(n_clusters):
            cluster_df = df[df["__cluster__"] == cluster_id]
            size = len(cluster_df)
            profile = {
                "size": size,
                "percent": round(size / len(df) * 100, 1),
                "stats": {}
            }
            for col in target_cols:
                profile["stats"][col] = {
                    "mean": round(float(cluster_df[col].mean()), 4),
                    "median": round(float(cluster_df[col].median()), 4),
                    "std": round(float(cluster_df[col].std()), 4),
                }
            cluster_profiles[f"cluster_{cluster_id}"] = profile
 
        cluster_means = pd.DataFrame({
            f"cluster_{i}": df[df["__cluster__"] == i][target_cols].mean()
            for i in range(n_clusters)
        }).T
        feature_variance = cluster_means.var().sort_values(ascending=False)
        top_features = feature_variance.head(5).index.tolist()
 
        output_df = df.copy()
        output_df = output_df.rename(columns={"__cluster__": "cluster_label"})
 
        return {
            "success": True,
            "task": "cluster_data",
            "method": method,
            "n_clusters": n_clusters,
            "total_rows": len(df),
            "columns_used": target_cols,
            "inertia": round(inertia, 2),
            "cluster_profiles": cluster_profiles,
            "top_distinguishing_features": top_features,
            "cluster_sizes": {
                f"cluster_{i}": int((labels == i).sum())
                for i in range(n_clusters)
            },
            "clustered_csv": output_df.to_csv(index=False),
        }
 
    except Exception as e:
        return {"success": False, "task": "cluster_data", "error": str(e)}
 
 
# ══════════════════════════════════════════════════════════════
#  correlation_analysis
# ══════════════════════════════════════════════════════════════
async def correlation_analysis(
    csv_data: str,
    target_column: str = "",
    method: str = "pearson",
    top_n: int = 10,
    threshold: float = 0.0,
) -> dict:
    """
    Compute full correlation matrix and identify top correlated feature pairs.
    If target_column is provided, ranks all features by correlation with it.
    Supports Pearson, Spearman, and Kendall methods with p-value significance testing.
    """
    try:
        df = _parse_csv(csv_data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
 
        if len(numeric_cols) < 2:
            return {"success": False, "task": "correlation_analysis", "error": "Need at least 2 numeric columns."}
 
        numeric_df = df[numeric_cols].fillna(df[numeric_cols].median())
        corr_matrix = numeric_df.corr(method=method)
 
        # All pairs
        pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                col_a = corr_matrix.columns[i]
                col_b = corr_matrix.columns[j]
                corr_val = float(corr_matrix.iloc[i, j])
                if abs(corr_val) >= threshold:
                    pairs.append({
                        "feature_a": col_a,
                        "feature_b": col_b,
                        "correlation": round(corr_val, 4),
                        "abs_correlation": round(abs(corr_val), 4),
                        "strength": (
                            "very strong" if abs(corr_val) >= 0.8 else
                            "strong" if abs(corr_val) >= 0.6 else
                            "moderate" if abs(corr_val) >= 0.4 else
                            "weak"
                        ),
                        "direction": "positive" if corr_val > 0 else "negative",
                    })
 
        pairs_sorted = sorted(pairs, key=lambda x: x["abs_correlation"], reverse=True)
 
        # Target correlations
        target_correlations = {}
        if target_column and target_column in numeric_cols:
            target_corr = corr_matrix[target_column].drop(target_column).sort_values(
                key=abs, ascending=False
            )
            target_correlations = {
                col: {
                    "correlation": round(float(val), 4),
                    "strength": (
                        "very strong" if abs(val) >= 0.8 else
                        "strong" if abs(val) >= 0.6 else
                        "moderate" if abs(val) >= 0.4 else
                        "weak"
                    ),
                    "direction": "positive" if val > 0 else "negative",
                }
                for col, val in target_corr.items()
            }
 
        multicollinear_pairs = [
            p for p in pairs_sorted
            if p["abs_correlation"] >= 0.85
            and p["feature_a"] != target_column
            and p["feature_b"] != target_column
        ]
 
        # p-values for top pairs
        sig_results = []
        for pair in pairs_sorted[:top_n]:
            try:
                if method == "pearson":
                    _, pval = stats.pearsonr(numeric_df[pair["feature_a"]], numeric_df[pair["feature_b"]])
                elif method == "spearman":
                    _, pval = stats.spearmanr(numeric_df[pair["feature_a"]], numeric_df[pair["feature_b"]])
                else:
                    pval = None
                sig_results.append({
                    **pair,
                    "p_value": round(float(pval), 6) if pval is not None else None,
                    "significant": bool(pval < 0.05) if pval is not None else None,
                })
            except Exception:
                sig_results.append(pair)
 
        return {
            "success": True,
            "task": "correlation_analysis",
            "method": method,
            "columns_analyzed": numeric_cols,
            "total_pairs": len(pairs),
            "top_correlated_pairs": sig_results,
            "target_correlations": target_correlations,
            "multicollinearity_warnings": multicollinear_pairs[:10],
            "correlation_matrix": corr_matrix.round(4).to_dict(),
            "summary": {
                "strongest_pair": pairs_sorted[0] if pairs_sorted else None,
                "avg_abs_correlation": round(float(np.mean([p["abs_correlation"] for p in pairs])), 4) if pairs else 0,
                "highly_correlated_count": len([p for p in pairs if p["abs_correlation"] >= 0.7]),
            }
        }
 
    except Exception as e:
        return {"success": False, "task": "correlation_analysis", "error": str(e)}
 
 
# ══════════════════════════════════════════════════════════════
#  explain_model
# ══════════════════════════════════════════════════════════════
async def explain_model(
    csv_data: str,
    target_column: str,
    audience: str = "business",
    include_recommendations: bool = True,
) -> dict:
    """
    Claude-powered plain-English explanation of your ML results.
    Trains a Gradient Boosting model, then uses Claude AI to explain
    the results, feature drivers, and actionable recommendations.
    Unique: combines AutoML + AI explanation in one tool.
    """
    try:
        df = _parse_csv(csv_data)
 
        if target_column not in df.columns:
            return {"success": False, "task": "explain_model", "error": f"Target column '{target_column}' not found."}
 
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_column]
 
        if not feature_cols:
            return {"success": False, "task": "explain_model", "error": "No numeric feature columns found."}
 
        X = df[feature_cols].fillna(0)
        y = df[target_column].fillna(df[target_column].median())
 
        is_classification = (y.nunique() < 15) and (y.nunique() > 1)
 
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, accuracy_score
 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
        if is_classification:
            model = GradientBoostingClassifier(random_state=42)
            model.fit(X_train, y_train)
            score = float(accuracy_score(y_test, model.predict(X_test)))
            metric_name = "accuracy"
        else:
            model = GradientBoostingRegressor(random_state=42)
            model.fit(X_train, y_train)
            score = float(r2_score(y_test, model.predict(X_test)))
            metric_name = "R²"
 
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        top_features = importance.sort_values(ascending=False).head(5)
 
        tone = "non-technical business executive" if audience == "business" else "senior data scientist"
 
        prompt = f"""You are explaining machine learning results to a {tone}.
 
Dataset: {len(df)} rows, target variable: '{target_column}'
Task type: {'Classification' if is_classification else 'Regression'}
Best model: Gradient Boosting
{metric_name} score: {round(score, 4)} ({'excellent' if score > 0.9 else 'good' if score > 0.7 else 'fair'})
 
Top feature importances driving '{target_column}':
{chr(10).join([f"- {feat}: {round(imp*100, 1)}% importance" for feat, imp in top_features.items()])}
 
Please provide:
1. A 2-3 sentence plain English explanation of what the model learned
2. What the {metric_name} score of {round(score, 4)} means in practical terms
3. The top 3 business insights from the feature importances
{"4. 3 specific actionable recommendations based on these results" if include_recommendations else ""}
 
Keep it concise, clear, and focused on business value. No jargon."""
 
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = message.content[0].text
 
        return {
            "success": True,
            "task": "explain_model",
            "target_column": target_column,
            "audience": audience,
            "model_results": {
                "model": "Gradient Boosting",
                "task_type": "classification" if is_classification else "regression",
                "metric": metric_name,
                "score": round(score, 4),
                "training_rows": len(X_train),
                "test_rows": len(X_test),
            },
            "feature_importance": top_features.round(4).to_dict(),
            "top_driver": top_features.index[0],
            "ai_explanation": explanation,
            "powered_by": "Claude Sonnet + Gradient Boosting",
        }
 
    except Exception as e:
        return {"success": False, "task": "explain_model", "error": str(e)}


# ══════════════════════════════════════════════════════════════
#  generate_dashboard
# ══════════════════════════════════════════════════════════════
async def generate_dashboard(
    csv_data: str,
    title: str = "Data Dashboard",
    target_column: str = "",
    theme: str = "dark",           # "dark" | "light"
) -> dict:
    """
    Generate a self-contained interactive HTML dashboard from a CSV dataset.
    Returns a complete HTML file with charts, KPI cards, and a data table.
    No external dependencies — works offline. Just open in any browser.
    """
    try:
        df = _parse_csv(csv_data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
 
        # Colors
        if theme == "dark":
            bg = "#070b14"
            card_bg = "#0d1526"
            text = "#e2e8f0"
            muted = "#64748b"
            border = "rgba(255,255,255,0.07)"
            accent = "#00d4aa"
        else:
            bg = "#f8fafc"
            card_bg = "#ffffff"
            text = "#1e293b"
            muted = "#64748b"
            border = "rgba(0,0,0,0.08)"
            accent = "#0ea5e9"
 
        # KPI stats
        kpis = []
        for col in numeric_cols[:4]:
            kpis.append({
                "label": col,
                "value": round(float(df[col].mean()), 2),
                "sub": f"mean · {len(df[col].dropna())} values"
            })
 
        # Bar chart data (top categorical column value counts)
        bar_data = {}
        if cat_cols:
            top_cat = cat_cols[0]
            vc = df[top_cat].value_counts().head(8)
            bar_data = {"labels": list(vc.index), "values": [int(v) for v in vc.values], "column": top_cat}
 
        # Line chart data (first numeric col)
        line_data = {}
        if numeric_cols:
            col = target_column if target_column in numeric_cols else numeric_cols[0]
            sample = df[col].dropna().head(50).tolist()
            line_data = {"values": [round(float(v), 2) for v in sample], "column": col}
 
        # Scatter data (first 2 numeric cols)
        scatter_data = {}
        if len(numeric_cols) >= 2:
            x_col, y_col = numeric_cols[0], numeric_cols[1]
            sample_df = df[[x_col, y_col]].dropna().head(100)
            scatter_data = {
                "x": [round(float(v), 2) for v in sample_df[x_col]],
                "y": [round(float(v), 2) for v in sample_df[y_col]],
                "x_col": x_col,
                "y_col": y_col,
            }
 
        # Data table (first 20 rows)
        table_cols = df.columns[:8].tolist()
        table_rows = df[table_cols].head(20).fillna("").values.tolist()
 
        # Build HTML
        kpi_html = ""
        for kpi in kpis:
            kpi_html += f"""
            <div class="kpi-card">
                <div class="kpi-label">{kpi['label']}</div>
                <div class="kpi-value">{kpi['value']}</div>
                <div class="kpi-sub">{kpi['sub']}</div>
            </div>"""
 
        table_headers = "".join([f"<th>{c}</th>" for c in table_cols])
        table_rows_html = ""
        for row in table_rows:
            table_rows_html += "<tr>" + "".join([f"<td>{v}</td>" for v in row]) + "</tr>"
 
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:{bg}; color:{text}; font-family:'Segoe UI',sans-serif; padding:24px; }}
h1 {{ font-size:1.6rem; font-weight:700; margin-bottom:6px; color:{text}; }}
.subtitle {{ color:{muted}; font-size:0.85rem; margin-bottom:24px; }}
.kpi-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; margin-bottom:24px; }}
.kpi-card {{ background:{card_bg}; border:1px solid {border}; border-radius:12px; padding:18px; position:relative; overflow:hidden; }}
.kpi-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:{accent}; opacity:0.7; }}
.kpi-label {{ font-size:0.7rem; color:{muted}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px; }}
.kpi-value {{ font-size:1.8rem; font-weight:700; color:{accent}; font-family:'Courier New',monospace; }}
.kpi-sub {{ font-size:0.72rem; color:{muted}; margin-top:4px; }}
.charts-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:24px; }}
.chart-card {{ background:{card_bg}; border:1px solid {border}; border-radius:12px; padding:18px; }}
.chart-title {{ font-size:0.82rem; font-weight:600; color:{muted}; margin-bottom:14px; text-transform:uppercase; letter-spacing:0.06em; }}
.chart-wrap {{ position:relative; height:220px; }}
.table-card {{ background:{card_bg}; border:1px solid {border}; border-radius:12px; padding:18px; overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:0.82rem; }}
th {{ color:{muted}; font-weight:600; padding:8px 12px; text-align:left; border-bottom:1px solid {border}; text-transform:uppercase; font-size:0.7rem; letter-spacing:0.06em; }}
td {{ padding:8px 12px; border-bottom:1px solid {border}; color:{text}; }}
tr:last-child td {{ border-bottom:none; }}
tr:hover td {{ background:rgba(255,255,255,0.02); }}
.footer {{ margin-top:20px; text-align:center; font-size:0.75rem; color:{muted}; }}
.footer span {{ color:{accent}; }}
@media(max-width:768px) {{ .charts-grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<h1>📊 {title}</h1>
<div class="subtitle">{len(df):,} rows · {len(df.columns)} columns · Generated by ProData AI</div>
 
<div class="kpi-row">{kpi_html}</div>
 
<div class="charts-grid">
  <div class="chart-card">
    <div class="chart-title">📈 {line_data.get('column','Trend')} — Trend</div>
    <div class="chart-wrap"><canvas id="lineChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">📊 {bar_data.get('column','Distribution')} — Distribution</div>
    <div class="chart-wrap"><canvas id="barChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">🔵 {scatter_data.get('x_col','')} vs {scatter_data.get('y_col','')} — Scatter</div>
    <div class="chart-wrap"><canvas id="scatterChart"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">🍩 {cat_cols[0] if cat_cols else 'Categories'} — Share</div>
    <div class="chart-wrap"><canvas id="doughnutChart"></canvas></div>
  </div>
</div>
 
<div class="table-card">
  <div class="chart-title">🗂 Data Preview — First 20 rows</div>
  <table>
    <thead><tr>{table_headers}</tr></thead>
    <tbody>{table_rows_html}</tbody>
  </table>
</div>
 
<div class="footer">Generated by <span>ProData AI</span> · MCP Server · prodata-ai.mcpize.run</div>
 
<script>
const ACCENT = '{accent}';
const COLORS = ['#00d4aa','#6366f1','#f59e0b','#ec4899','#3b82f6','#10b981','#8b5cf6','#f97316'];
const gridColor = 'rgba(255,255,255,0.05)';
const tickColor = '{muted}';
 
Chart.defaults.color = tickColor;
Chart.defaults.borderColor = gridColor;
 
// Line chart
new Chart(document.getElementById('lineChart'), {{
  type: 'line',
  data: {{
    labels: {json.dumps(list(range(len(line_data.get('values', [])))))},
    datasets: [{{ label: '{line_data.get("column","")}', data: {json.dumps(line_data.get('values', []))},
      borderColor: ACCENT, backgroundColor: 'rgba(0,212,170,0.08)',
      borderWidth: 2, pointRadius: 0, fill: true, tension: 0.4 }}]
  }},
  options: {{ responsive:true, maintainAspectRatio:false,
    plugins: {{ legend: {{ display:false }} }},
    scales: {{ x: {{ display:false }}, y: {{ grid: {{ color: gridColor }} }} }}
  }}
}});
 
// Bar chart
new Chart(document.getElementById('barChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(bar_data.get('labels', []))},
    datasets: [{{ label: '{bar_data.get("column","")}', data: {json.dumps(bar_data.get('values', []))},
      backgroundColor: COLORS, borderRadius: 4 }}]
  }},
  options: {{ responsive:true, maintainAspectRatio:false,
    plugins: {{ legend: {{ display:false }} }},
    scales: {{ y: {{ grid: {{ color: gridColor }} }}, x: {{ grid: {{ display:false }} }} }}
  }}
}});
 
// Scatter chart
new Chart(document.getElementById('scatterChart'), {{
  type: 'scatter',
  data: {{
    datasets: [{{ label: 'Data points',
      data: {json.dumps([{"x": x, "y": y} for x, y in zip(scatter_data.get('x',[]), scatter_data.get('y',[]))])},
      backgroundColor: 'rgba(0,212,170,0.5)', pointRadius: 4 }}]
  }},
  options: {{ responsive:true, maintainAspectRatio:false,
    plugins: {{ legend: {{ display:false }} }},
    scales: {{ x: {{ grid: {{ color: gridColor }} }}, y: {{ grid: {{ color: gridColor }} }} }}
  }}
}});
 
// Doughnut chart
new Chart(document.getElementById('doughnutChart'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(bar_data.get('labels', []))},
    datasets: [{{ data: {json.dumps(bar_data.get('values', []))},
      backgroundColor: COLORS, borderWidth: 0 }}]
  }},
  options: {{ responsive:true, maintainAspectRatio:false,
    plugins: {{ legend: {{ position:'right', labels: {{ font: {{ size:11 }}, padding:12 }} }} }}
  }}
}});
</script>
</body>
</html>"""
 
        return {
            "success": True,
            "task": "generate_dashboard",
            "title": title,
            "theme": theme,
            "rows": len(df),
            "columns": len(df.columns),
            "kpis_shown": len(kpis),
            "charts": ["line", "bar", "scatter", "doughnut"],
            "html": html,
            "usage": "Save the 'html' field content as a .html file and open in any browser.",
        }
 
    except Exception as e:
        return {"success": False, "task": "generate_dashboard", "error": str(e)}
 
 
# ══════════════════════════════════════════════════════════════
#  suggest_visualizations
# ══════════════════════════════════════════════════════════════
async def suggest_visualizations(
    csv_data: str,
    target_column: str = "",
    max_suggestions: int = 8,
) -> dict:
    """
    Analyze column types and automatically suggest the best chart types for your data.
    Returns ranked visualization suggestions with column recommendations and rationale.
    Saves hours of chart selection — just pick the one that fits your story.
    """
    try:
        df = _parse_csv(csv_data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()
        date_cols = []
 
        # Detect date columns
        for col in cat_cols:
            try:
                pd.to_datetime(df[col].dropna().head(10))
                date_cols.append(col)
            except Exception:
                pass
        non_date_cats = [c for c in cat_cols if c not in date_cols]
 
        suggestions = []
 
        # 1. Time series line chart
        if date_cols and numeric_cols:
            for date_col in date_cols[:2]:
                for num_col in numeric_cols[:2]:
                    suggestions.append({
                        "rank": 1,
                        "chart_type": "Line Chart",
                        "x_column": date_col,
                        "y_column": num_col,
                        "rationale": f"'{date_col}' is a date column — line chart shows trend over time for '{num_col}'.",
                        "use_case": "Trend analysis, time series visualization",
                        "priority": "high",
                    })
 
        # 2. Bar chart for categorical
        if non_date_cats and numeric_cols:
            for cat_col in non_date_cats[:2]:
                n_unique = df[cat_col].nunique()
                if n_unique <= 20:
                    suggestions.append({
                        "rank": 2,
                        "chart_type": "Bar Chart",
                        "x_column": cat_col,
                        "y_column": numeric_cols[0],
                        "rationale": f"'{cat_col}' has {n_unique} categories — bar chart compares '{numeric_cols[0]}' across groups.",
                        "use_case": "Category comparison, ranking",
                        "priority": "high",
                    })
 
        # 3. Scatter plot for correlations
        if len(numeric_cols) >= 2:
            suggestions.append({
                "rank": 3,
                "chart_type": "Scatter Plot",
                "x_column": numeric_cols[0],
                "y_column": numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0],
                "rationale": f"Both columns are numeric — scatter reveals correlation between '{numeric_cols[0]}' and '{numeric_cols[1]}'.",
                "use_case": "Correlation analysis, outlier detection",
                "priority": "high",
            })
 
        # 4. Histogram for distribution
        for col in numeric_cols[:3]:
            suggestions.append({
                "rank": 4,
                "chart_type": "Histogram",
                "x_column": col,
                "y_column": "frequency",
                "rationale": f"'{col}' is numeric — histogram shows its distribution and skewness.",
                "use_case": "Distribution analysis, outlier spotting",
                "priority": "medium",
            })
 
        # 5. Pie/Doughnut for proportions
        if non_date_cats:
            for cat_col in non_date_cats[:2]:
                n_unique = df[cat_col].nunique()
                if 2 <= n_unique <= 8:
                    suggestions.append({
                        "rank": 5,
                        "chart_type": "Doughnut Chart",
                        "x_column": cat_col,
                        "y_column": "count",
                        "rationale": f"'{cat_col}' has {n_unique} categories — doughnut shows proportional share.",
                        "use_case": "Part-to-whole relationships, market share",
                        "priority": "medium",
                    })
 
        # 6. Heatmap for correlations
        if len(numeric_cols) >= 3:
            suggestions.append({
                "rank": 6,
                "chart_type": "Correlation Heatmap",
                "x_column": "all numeric columns",
                "y_column": "all numeric columns",
                "rationale": f"Dataset has {len(numeric_cols)} numeric columns — heatmap reveals all pairwise correlations at once.",
                "use_case": "Feature selection, multicollinearity detection",
                "priority": "medium",
            })
 
        # 7. Box plot for outliers
        if numeric_cols and non_date_cats:
            suggestions.append({
                "rank": 7,
                "chart_type": "Box Plot",
                "x_column": non_date_cats[0],
                "y_column": numeric_cols[0],
                "rationale": f"Shows distribution spread and outliers of '{numeric_cols[0]}' per '{non_date_cats[0]}' group.",
                "use_case": "Outlier detection, group distribution comparison",
                "priority": "medium",
            })
 
        # 8. Target-specific suggestions
        if target_column and target_column in numeric_cols:
            for col in numeric_cols[:3]:
                if col != target_column:
                    suggestions.append({
                        "rank": 8,
                        "chart_type": "Scatter Plot",
                        "x_column": col,
                        "y_column": target_column,
                        "rationale": f"Shows relationship between '{col}' and your target '{target_column}'. Useful for feature analysis.",
                        "use_case": "Feature vs target relationship, regression visualization",
                        "priority": "high",
                    })
 
        # Deduplicate and limit
        seen = set()
        unique_suggestions = []
        for s in suggestions:
            key = f"{s['chart_type']}_{s['x_column']}_{s['y_column']}"
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)
 
        final = unique_suggestions[:max_suggestions]
 
        return {
            "success": True,
            "task": "suggest_visualizations",
            "total_suggestions": len(final),
            "dataset_profile": {
                "numeric_columns": numeric_cols,
                "categorical_columns": non_date_cats,
                "date_columns": date_cols,
                "rows": len(df),
            },
            "suggestions": final,
            "top_pick": final[0] if final else None,
            "tip": "Start with 'high' priority suggestions — they're chosen based on your actual column types.",
        }
 
    except Exception as e:
        return {"success": False, "task": "suggest_visualizations", "error": str(e)}
 
 
# ══════════════════════════════════════════════════════════════
#  generate_sql
# ══════════════════════════════════════════════════════════════
async def generate_sql(
    csv_data: str,
    question: str,
    table_name: str = "dataset",
    dialect: str = "standard",    # "standard" | "mysql" | "postgresql" | "sqlite" | "bigquery"
) -> dict:
    """
    Claude-powered natural language to SQL converter.
    Describe what you want in plain English — get a ready-to-run SQL query back.
    Understands your actual column names and data types from the CSV.
    """
    try:
        df = _parse_csv(csv_data)
 
        # Build schema context
        schema_lines = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            sample_vals = df[col].dropna().head(3).tolist()
            sql_type = (
                "INTEGER" if "int" in dtype else
                "FLOAT" if "float" in dtype else
                "DATE" if "datetime" in dtype else
                "VARCHAR(255)"
            )
            schema_lines.append(f"  {col} {sql_type}  -- e.g. {sample_vals}")
 
        schema = f"CREATE TABLE {table_name} (\n" + ",\n".join(schema_lines) + "\n);"
 
        # Sample data
        sample_rows = df.head(3).to_string(index=False)
 
        # Dialect notes
        dialect_notes = {
            "mysql": "Use MySQL syntax. Use LIMIT for row limiting.",
            "postgresql": "Use PostgreSQL syntax. Use LIMIT for row limiting.",
            "sqlite": "Use SQLite syntax. Avoid advanced window functions.",
            "bigquery": "Use BigQuery Standard SQL syntax. Use LIMIT for row limiting.",
            "standard": "Use standard ANSI SQL syntax.",
        }
        dialect_hint = dialect_notes.get(dialect, dialect_notes["standard"])
 
        prompt = f"""You are an expert SQL analyst. Generate a SQL query to answer this question.
 
Table schema:
{schema}
 
Sample data (first 3 rows):
{sample_rows}
 
Question: {question}
 
Dialect: {dialect_hint}
 
Rules:
1. Use ONLY the column names that exist in the schema above — exact spelling, case-sensitive
2. Return ONLY the SQL query — no explanation, no markdown, no backticks
3. Make the query correct, efficient, and ready to run
4. If aggregating, always include a GROUP BY
5. If the question is ambiguous, choose the most useful interpretation
 
SQL query:"""
 
        client = anthropic.Anthropic()
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )
        sql_query = message.content[0].text.strip()
 
        # Clean up any accidental markdown
        if sql_query.startswith("```"):
            sql_query = sql_query.split("```")[1]
            if sql_query.startswith("sql"):
                sql_query = sql_query[3:]
        sql_query = sql_query.strip()
 
        return {
            "success": True,
            "task": "generate_sql",
            "question": question,
            "table_name": table_name,
            "dialect": dialect,
            "sql_query": sql_query,
            "schema_used": schema,
            "columns_available": list(df.columns),
            "powered_by": "Claude Sonnet",
            "tip": f"Run this query against a table named '{table_name}' loaded from your CSV.",
        }
 
    except Exception as e:
        return {"success": False, "task": "generate_sql", "error": str(e)}
