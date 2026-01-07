"""
Global Configuration and Utility Functions
Shared across the FNOL Claims Intelligence System
"""
import os
import sys
from pathlib import Path 
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')
import pins
from vetiver import VetiverModel
from huggingface_hub import snapshot_download

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ============================================================================
# APPLICATION CONFIGURATION
# ============================================================================

APP_CONFIG = {
    "app_name": "Advanced FNOL Claims Intelligence System",
    "company": "Admiral Group",
    "version": "1.0.0",
    "department": "Claims Analytics & Operations",
    "primary_color": "#003B5C",  # Admiral Navy Blue
    "secondary_color": "#FF6B35",  # Admiral Orange
    "success_color": "#28A745",
    "warning_color": "#FFC107",
    "danger_color": "#DC3545",
    "info_color": "#17A2B8"
}

# Model configuration
MODEL_CONFIG = {
    "huggingface_repo": "your-username/fnol-claims-model",
    "pin_board_url": "model_pins_board",
    "available_models": ["gradient_boosting", "random_forest", "xgboost","linear_regression"],
    "target_variable": "Ultimate_Claim_Amount",
    "prediction_confidence_threshold": 0.75
}

# Business thresholds for Admiral Group
BUSINESS_THRESHOLDS = {
    "low_severity": 5000,
    "medium_severity": 15000,
    "high_severity": 30000,
    "critical_severity": 50000,
    "fnol_delay_threshold_hours": 24,
    "settlement_target_days": 30,
    "reserve_buffer_percentage": 15
}

# Feature importance mapping for business context
FEATURE_BUSINESS_CONTEXT = {
    "Estimated_Claim_Amount": {
        "name": "Initial FNOL Estimate",
        "business_impact": "Baseline cost assessment",
        "actionable_insight": "Review estimation accuracy"
    },
    "Driver_Age": {
        "name": "Driver Age",
        "business_impact": "Risk profile indicator",
        "actionable_insight": "Monitor age-based risk patterns"
    },
    "Vehicle_Age": {
        "name": "Vehicle Age",
        "business_impact": "Repair cost predictor",
        "actionable_insight": "Assess depreciation impact"
    },
    "License_Age": {
        "name": "License Experience",
        "business_impact": "Driver experience level",
        "actionable_insight": "Experience-based risk assessment"
    },
    "Claim_Type": {
        "name": "Claim Category",
        "business_impact": "Cost pattern driver",
        "actionable_insight": "Type-specific handling protocols"
    },
    "FNOL_Delay_Hours": {
        "name": "Reporting Delay",
        "business_impact": "Cost escalation indicator",
        "actionable_insight": "Investigate delayed reporting"
    },
    "Weather_Condition": {
        "name": "Weather Conditions",
        "business_impact": "Severity amplifier",
        "actionable_insight": "Weather-based risk adjustment"
    },
    "Traffic_Condition": {
        "name": "Traffic Density",
        "business_impact": "Accident severity context",
        "actionable_insight": "Traffic pattern analysis"
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_currency(amount: float) -> str:
    """Format number as GBP currency"""
    return f"£{amount:,.2f}"

def format_percentage(value: float) -> str:
    """Format number as percentage"""
    return f"{value:.1f}%"

def calculate_severity_level(amount: float) -> Tuple[str, str]:
    """
    Determine claim severity level and color
    Returns: (severity_level, color)
    """
    if amount < BUSINESS_THRESHOLDS["low_severity"]:
        return "Low", APP_CONFIG["success_color"]
    elif amount < BUSINESS_THRESHOLDS["medium_severity"]:
        return "Medium", APP_CONFIG["info_color"]
    elif amount < BUSINESS_THRESHOLDS["high_severity"]:
        return "High", APP_CONFIG["warning_color"]
    else:
        return "Critical", APP_CONFIG["danger_color"]

def calculate_reserve_recommendation(predicted_amount: float) -> float:
    """Calculate recommended reserve with buffer"""
    buffer = BUSINESS_THRESHOLDS["reserve_buffer_percentage"] / 100
    return predicted_amount * (1 + buffer)

def calculate_prediction_variance(predictions: List[float]) -> Dict:
    """Calculate statistical measures of prediction confidence"""
    if not predictions or len(predictions) < 2:
        return {"mean": 0, "std": 0, "cv": 0, "confidence": "Low"}
    
    mean_pred = np.mean(predictions)
    std_pred = np.std(predictions)
    cv = (std_pred / mean_pred * 100) if mean_pred > 0 else 0
    
    # Confidence based on coefficient of variation
    if cv < 10:
        confidence = "High"
    elif cv < 25:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return {
        "mean": mean_pred,
        "std": std_pred,
        "cv": cv,
        "confidence": confidence
    }

def calculate_business_metrics(df: pd.DataFrame) -> Dict:
    """Calculate key business metrics from predictions"""
    if df.empty:
        return {}
    
    metrics = {
        "total_claims": len(df),
        "total_exposure": df["Ultimate_Claim_Amount"].sum() if "Ultimate_Claim_Amount" in df.columns else 0,
        "average_claim": df["Ultimate_Claim_Amount"].mean() if "Ultimate_Claim_Amount" in df.columns else 0,
        "median_claim": df["Ultimate_Claim_Amount"].median() if "Ultimate_Claim_Amount" in df.columns else 0,
        "max_claim": df["Ultimate_Claim_Amount"].max() if "Ultimate_Claim_Amount" in df.columns else 0,
        "severity_distribution": {}
    }
    
    # Calculate severity distribution
    if "Ultimate_Claim_Amount" in df.columns:
        for _, row in df.iterrows():
            severity, _ = calculate_severity_level(row["Ultimate_Claim_Amount"])
            metrics["severity_distribution"][severity] = metrics["severity_distribution"].get(severity, 0) + 1
    
    return metrics

def generate_actionable_insights(prediction: float, features: Dict) -> List[Dict]:
    """Generate actionable business insights based on prediction and features"""
    insights = []
    severity, _ = calculate_severity_level(prediction)
    
    # Severity-based insights
    if severity == "Critical":
        insights.append({
            "type": "alert",
            "title": "High-Value Claim Alert",
            "message": f"Predicted amount ({format_currency(prediction)}) exceeds critical threshold. Immediate senior review recommended.",
            "action": "Escalate to claims manager and fraud investigation team",
            "priority": "High"
        })
    
    # FNOL delay insights
    if "FNOL_Delay_Hours" in features and features["FNOL_Delay_Hours"] > BUSINESS_THRESHOLDS["fnol_delay_threshold_hours"]:
        insights.append({
            "type": "warning",
            "title": "Delayed FNOL Reporting",
            "message": f"Claim reported {features['FNOL_Delay_Hours']:.0f} hours after incident.",
            "action": "Request detailed incident timeline; check for fraud indicators",
            "priority": "Medium"
        })
    
    # Age-based risk insights
    if "Driver_Age" in features:
        if features["Driver_Age"] < 25:
            insights.append({
                "type": "info",
                "title": "Young Driver Risk Profile",
                "message": "Driver age suggests higher risk category.",
                "action": "Apply enhanced scrutiny to damage assessment",
                "priority": "Medium"
            })
        elif features["Driver_Age"] > 70:
            insights.append({
                "type": "info",
                "title": "Senior Driver Consideration",
                "message": "Elderly driver may require additional support.",
                "action": "Ensure clear communication and customer support",
                "priority": "Low"
            })
    
    # Vehicle age insights
    if "Vehicle_Age" in features and features["Vehicle_Age"] > 15:
        insights.append({
            "type": "warning",
            "title": "Older Vehicle Assessment",
            "message": f"Vehicle age ({features['Vehicle_Age']:.0f} years) may affect repair economics.",
            "action": "Evaluate total loss vs. repair cost-effectiveness",
            "priority": "Medium"
        })
    
    # Weather-related insights
    if "Weather_Condition" in features and features["Weather_Condition"] in ["Rainy", "Foggy", "Snowy"]:
        insights.append({
            "type": "info",
            "title": "Adverse Weather Conditions",
            "message": f"Incident occurred in {features['Weather_Condition']} conditions.",
            "action": "Document weather impact on incident severity",
            "priority": "Low"
        })
    
    return insights

def create_plotly_theme() -> Dict:
    """Create consistent Plotly theme matching Admiral branding"""
    return {
        "layout": {
            "font": {"family": "Arial, sans-serif", "size": 12, "color": APP_CONFIG["primary_color"]},
            "plot_bgcolor": "#FFFFFF",
            "paper_bgcolor": "#F8F9FA",
            "title": {"font": {"size": 16, "color": APP_CONFIG["primary_color"]}},
            "xaxis": {"gridcolor": "#E9ECEF", "showline": True, "linecolor": "#DEE2E6"},
            "yaxis": {"gridcolor": "#E9ECEF", "showline": True, "linecolor": "#DEE2E6"}
        }
    }

def create_severity_gauge(predicted_amount: float) -> go.Figure:
    """Create gauge chart showing claim severity"""
    severity, color = calculate_severity_level(predicted_amount)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted_amount,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Predicted Claim Amount", 'font': {'size': 16}},
        delta={'reference': BUSINESS_THRESHOLDS["medium_severity"]},
        gauge={
            'axis': {'range': [None, BUSINESS_THRESHOLDS["critical_severity"]]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, BUSINESS_THRESHOLDS["low_severity"]], 'color': "#E8F5E9"},
                {'range': [BUSINESS_THRESHOLDS["low_severity"], BUSINESS_THRESHOLDS["medium_severity"]], 'color': "#FFF9C4"},
                {'range': [BUSINESS_THRESHOLDS["medium_severity"], BUSINESS_THRESHOLDS["high_severity"]], 'color': "#FFE0B2"},
                {'range': [BUSINESS_THRESHOLDS["high_severity"], BUSINESS_THRESHOLDS["critical_severity"]], 'color': "#FFCDD2"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': BUSINESS_THRESHOLDS["high_severity"]
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        paper_bgcolor="#F8F9FA",
        font={'color': APP_CONFIG["primary_color"], 'family': "Arial"}
    )
    
    return fig

def validate_input_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate uploaded data for required columns and data quality"""
    required_columns = [
        "Policy_ID", "Claim_ID", "Estimated_Claim_Amount", 
        "Driver_Age", "Vehicle_Age", "Claim_Type"
    ]
    
    errors = []
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")
    
    # Check for empty dataframe
    if df.empty:
        errors.append("Uploaded file contains no data")
    
    # Check for null values in critical columns
    if not df.empty:
        for col in required_columns:
            if col in df.columns and df[col].isnull().sum() > 0:
                null_pct = (df[col].isnull().sum() / len(df)) * 100
                if null_pct > 10:
                    errors.append(f"{col} has {null_pct:.1f}% missing values (threshold: 10%)")
    
    return len(errors) == 0, errors

def calculate_roi_metrics(predictions_df: pd.DataFrame) -> Dict:
    """Calculate ROI and business value metrics"""
    if predictions_df.empty:
        return {}
    
    # Simulate cost savings from accurate predictions
    total_predicted = predictions_df["Ultimate_Claim_Amount"].sum() if "Ultimate_Claim_Amount" in predictions_df.columns else 0
    
    # Conservative estimates
    reserve_optimization_savings = total_predicted * 0.03  # 3% reduction in excess reserves
    fraud_prevention_savings = total_predicted * 0.02  # 2% fraud detection improvement
    operational_efficiency_savings = len(predictions_df) * 50  # £50 per claim in handling efficiency
    
    return {
        "total_exposure": total_predicted,
        "reserve_optimization": reserve_optimization_savings,
        "fraud_prevention": fraud_prevention_savings,
        "operational_efficiency": operational_efficiency_savings,
        "total_savings": reserve_optimization_savings + fraud_prevention_savings + operational_efficiency_savings
    }

def get_modeling_data(df, numeric_features, categorical_features):

    # Filter to only include features that exist in the current dataframe
    available_numeric = [f for f in numeric_features if f in df.columns]
    available_categorical = [f for f in categorical_features if f in df.columns]
    all_features = available_numeric + available_categorical 
        
    # Return the transformed array (prediction format)
    return df[all_features].copy()

def load_model(url,board_name):
    # connecting to hugingface
    temp_dir = tempfile.mkdtemp()
    print("Downloading model from Hugging Face...")
    model_path = snapshot_download(repo_id = url,
                                   cache_dir = temp_dir,
                                   local_dir = os.path.join(temp_dir,board_name),
                                   local_dir_use_symlinks = False)
    return model_path

def run_predictions(transformed_array, model_path, model_name):
    # 1. Connect to board and retrieve model
    # Create a temporary pins board from the downloaded folder
    temp_board = pins.board_folder(model_path, allow_pickle_read=True)

    # board = pins.board_folder(board_name, allow_pickle_read=True)
    v = VetiverModel.from_pin(temp_board, model_name)

    # 2. Predict
    return v.model.predict(transformed_array)

# ============================================================================
# DATA PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_claim_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess claim data for prediction"""
    df = df.copy()
    
    # Calculate derived features
    if "Accident_Date" in df.columns and "FNOL_Date" in df.columns:
        df["FNOL_Delay_Hours"] = (
            pd.to_datetime(df["FNOL_Date"]) - pd.to_datetime(df["Accident_Date"])
        ).dt.total_seconds() / 3600
    
    if "Date_of_Birth" in df.columns:
        df["Driver_Age"] = (datetime.now() - pd.to_datetime(df["Date_of_Birth"])).dt.days / 365.25
    
    if "Full_License_Issue_Date" in df.columns:
        df["License_Age"] = (datetime.now() - pd.to_datetime(df["Full_License_Issue_Date"])).dt.days / 365.25
    
    if "Vehicle_Year" in df.columns:
        df["Vehicle_Age"] = datetime.now().year - df["Vehicle_Year"]
    
    return df

def get_feature_statistics(df: pd.DataFrame) -> Dict:
    """Calculate descriptive statistics for features"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": df[col].mean(),
            "median": df[col].median(),
            "std": df[col].std(),
            "min": df[col].min(),
            "max": df[col].max(),
            "q25": df[col].quantile(0.25),
            "q75": df[col].quantile(0.75)
        }
    
    return stats