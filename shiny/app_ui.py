"""
User Interface Design
Advanced FNOL Claims Intelligence System
"""

from pathlib import Path
from shiny import ui
from global_py import APP_CONFIG, MODEL_CONFIG
import shinyswatch
from shinywidgets import output_widget

# ============================================================================
# UI LAYOUT
# ============================================================================
# Define the path to your www directory
www_dir = Path(__file__).parent / "www"

app_ui = ui.page_navbar(
    
    # Custom CSS ======================================
    ui.head_content(
        ui.tags.link(rel="stylesheet", href="custom.css"),
        ui.tags.style("""
            :root {
                --admiral-navy: #003B5C;
                --admiral-orange: #FF6B35;
            }
            
            .navbar {
                background-color: var(--admiral-navy) !important;
            }
            
            .metric-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .metric-value {
                font-size: 2.5rem;
                font-weight: bold;
                margin: 10px 0;
            }
            
            .metric-label {
                font-size: 0.9rem;
                opacity: 0.9;
                text-transform: uppercase;
                letter-spacing: 1px;
            }
            
            .insight-card {
                border-left: 4px solid var(--admiral-orange);
                padding: 15px;
                margin: 10px 0;
                background: #f8f9fa;
                border-radius: 5px;
            }
            
            .insight-title {
                font-weight: bold;
                color: var(--admiral-navy);
                margin-bottom: 8px;
            }
            
            .card-enhanced {
                border: none;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                border-radius: 8px;
                transition: transform 0.2s;
            }
            
            .card-enhanced:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            }
            
            .alert-high {
                background-color: #ffe6e6;
                border-left: 4px solid #dc3545;
            }
            
            .alert-medium {
                background-color: #fff3cd;
                border-left: 4px solid #ffc107;
            }
            
            .alert-low {
                background-color: #d1ecf1;
                border-left: 4px solid #17a2b8;
            }
            
            .btn-admiral {
                background-color: var(--admiral-orange);
                color: white;
                border: none;
                padding: 10px 25px;
                border-radius: 5px;
                font-weight: 600;
                transition: all 0.3s;
            }
            
            .btn-admiral:hover {
                background-color: #e55a2b;
                transform: translateY(-1px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
        """)
    ),
        
    # ========================================================================
    # TAB 1: DASHBOARD & OVERVIEW
    # ========================================================================
    ui.nav_panel(
        "📊 Dashboard",
        ui.layout_sidebar(
            ui.sidebar(
                ui.div(
                    {"style": "text-align: center; padding: 20px;"},
                    ui.img(src="Victor_Logo2.png", style="max-width: 150px; margin-bottom: 20px;"),
                    ui.h4("Admiral Group", style="color: var(--admiral-navy);"),
                    ui.p("Claims Intelligence Platform", style="font-size: 0.9rem; color: #666;")
                ),
                ui.hr(),
                ui.input_select(
                    "dashboard_metric",
                    "Key Metric View",
                    choices={
                        "exposure": "Total Exposure",
                        "volume": "Claim Volume",
                        "severity": "Severity Distribution",
                        "efficiency": "Operational Efficiency"
                    }
                ),
                ui.input_date_range(
                    "dashboard_date_range",
                    "Analysis Period",
                    start="2024-01-01"
                ),
                ui.hr(),
                ui.div(
                    {"class": "metric-card"},
                    ui.div("System Status", {"class": "metric-label"}),
                    ui.output_text("system_status"),
                    ui.div("Models Loaded", {"class": "metric-label", "style": "margin-top: 15px;"}),
                    ui.output_text("models_count")
                ),
                width=300
            ),
            
            # Main dashboard content
            ui.div(
                ui.row(
                    ui.column(
                        3,
                        ui.div(
                            {"class": "card card-enhanced"},
                            ui.card_header("Total Claims Processed"),
                            ui.card_body(
                                ui.output_text("total_claims_processed", container=ui.h2)
                            )
                        )
                    ),
                    ui.column(
                        3,
                        ui.div(
                            {"class": "card card-enhanced"},
                            ui.card_header("Total Exposure"),
                            ui.card_body(
                                ui.output_text("total_exposure_amount", container=ui.h2)
                            )
                        )
                    ),
                    ui.column(
                        3,
                        ui.div(
                            {"class": "card card-enhanced"},
                            ui.card_header("Average Claim Value"),
                            ui.card_body(
                                ui.output_text("average_claim_value", container=ui.h2)
                            )
                        )
                    ),
                    ui.column(
                        3,
                        ui.div(
                            {"class": "card card-enhanced"},
                            ui.card_header("High Severity Claims"),
                            ui.card_body(
                                ui.output_text("high_severity_count", container=ui.h2)
                            )
                        )
                    )
                ),
                
                ui.row(
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Claim Severity Distribution"),
                            output_widget("severity_distribution_plot")
                        )
                    ),
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Predicted vs. Estimated Amounts"),
                            output_widget("prediction_accuracy_plot")
                        )
                    )
                ),
                
                ui.row(
                    ui.column(
                        12,
                        ui.card(
                            ui.card_header("Business Intelligence Insights"),
                            ui.output_ui("business_insights_dashboard")
                        )
                    )
                ),
                
                ui.row(
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Claims by Type"),
                            output_widget("claims_by_type_plot")
                        )
                    ),
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("ROI Metrics"),
                            output_widget("roi_metrics_plot")
                        )
                    )
                )
            )
        )
    ),
    
    # ========================================================================
    # TAB 2: SINGLE CLAIM PREDICTION
    # ========================================================================
    ui.nav_panel(
        "🔍 Single Claim Analysis",
        ui.layout_sidebar(
            ui.sidebar(
                ui.h4("Claim Details", style="color: var(--admiral-navy);"),
                ui.hr(),
                
                ui.input_text("claim_id", "Claim ID", placeholder="CLM-2024-00001"),
                ui.input_text("policy_id", "Policy ID", placeholder="POL-2024-00001"),
                
                ui.hr(),
                ui.h5("Claim Information"),
                
                ui.input_numeric("estimated_amount", "Initial Estimate (£)", value=5000, min=0, max=500000),
                
                ui.input_select(
                    "claim_type",
                    "Claim Type",
                    choices={
                        "Third-Party Damage": "Third-Party Damage",
                        "Collision": "Collision",
                        "Windshield Damage": "Windshield Damage",
                        "Theft": "Theft",
                        "Fire": "Fire"
                    }
                ),
                
                ui.input_numeric("driver_age", "Driver Age", value=35, min=17, max=100),
                ui.input_numeric("license_age", "License Age (years)", value=10, min=0, max=80),
                ui.input_numeric("vehicle_age", "Vehicle Age (years)", value=5, min=0, max=30),
                
                ui.input_select(
                    "weather_condition",
                    "Weather Condition",
                    choices=["Clear", "Rainy", "Foggy", "Snowy"]
                ),
                
                ui.input_select(
                    "traffic_condition",
                    "Traffic Condition",
                    choices=["Light", "Moderate", "Heavy"]
                ),
                
                ui.input_numeric("fnol_delay_hours", "FNOL Delay (hours)", value=2, min=0, max=720),
                
                ui.hr(),
                ui.input_action_button("predict_btn", "Generate Prediction", class_="btn-admiral", style="width: 100%;"),
                
                width=350
            ),
            
            # Prediction results
            ui.div(
                ui.row(
                    ui.column(
                        12,
                        ui.div(
                            {"style": "background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 20px;"},
                            ui.h2("Claim Prediction Results", style="margin-bottom: 20px;"),
                            ui.row(
                                ui.column(
                                    4,
                                    ui.div(
                                        ui.div("Predicted Amount", style="font-size: 0.9rem; opacity: 0.9;"),
                                        ui.output_text("prediction_amount", container=ui.h1)
                                    )
                                ),
                                ui.column(
                                    4,
                                    ui.div(
                                        ui.div("Severity Level", style="font-size: 0.9rem; opacity: 0.9;"),
                                        ui.output_text("prediction_severity", container=ui.h1)
                                    )
                                ),
                                ui.column(
                                    4,
                                    ui.div(
                                        ui.div("Recommended Reserve", style="font-size: 0.9rem; opacity: 0.9;"),
                                        ui.output_text("recommended_reserve", container=ui.h1)
                                    )
                                )
                            )
                        )
                    )
                ),
                
                ui.row(
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Severity Gauge"),
                            ui.output_ui("severity_gauge_plot")
                        )
                    ),
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Feature Importance (SHAP)"),
                            output_widget("shap_plot")
                        )
                    )
                ),
                
                ui.row(
                    ui.column(
                        12,
                        ui.card(
                            ui.card_header("🎯 Actionable Business Insights"),
                            ui.output_ui("actionable_insights")
                        )
                    )
                ),
                
                ui.row(
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Risk Factors Analysis"),
                            ui.output_ui("risk_factors_analysis")
                        )
                    ),
                    ui.column(
                        6,
                        ui.card(
                            ui.card_header("Recommended Actions"),
                            ui.output_ui("recommended_actions")
                        )
                    )
                )
            )
        )
    ),
    
    # ========================================================================
    # TAB 3: BATCH PREDICTION & UPLOAD
    # ========================================================================
    ui.nav_panel(
        "📁 Batch Prediction",
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Upload Claims Data for Batch Processing"),
                    ui.card_body(
                        ui.row(
                            ui.column(
                                6,
                                ui.input_file("file_upload", "Upload CSV File", accept=[".csv"], multiple=False),
                                ui.tags.small("Required columns: Policy_ID, Claim_ID, Estimated_Claim_Amount, Driver_Age, Vehicle_Age, Claim_Type"),
                                ui.hr(),
                                ui.input_action_button("process_batch_btn", "Process Batch", class_="btn-admiral"),
                                ui.download_button("download_predictions", "Download Results", class_="btn-admiral", style="margin-left: 10px;")
                            ),
                            ui.column(
                                6,
                                ui.h5("Data Validation Status"),
                                ui.output_ui("validation_status"),
                                ui.hr(),
                                ui.output_text("uploaded_rows_count")
                            )
                        )
                    )
                )
            )
        ),
        
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Batch Processing Results"),
                    ui.output_data_frame("batch_results_table")
                )
            )
        ),
        
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Batch Statistics"),
                    output_widget("batch_statistics_plot")
                )
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Predicted Amounts Distribution"),
                    output_widget("batch_distribution_plot")
                )
            )
        )
    ),
    
    # ========================================================================
    # TAB 4: MODEL MANAGEMENT
    # ========================================================================
    ui.nav_panel(
        "🤖 Model Management",
        ui.row(
            ui.column(
                4,
                ui.card(
                    ui.card_header("Model Selection"),
                    ui.card_body(
                        ui.input_select(
                            "selected_model",
                            "Active Model",
                            choices=MODEL_CONFIG["available_models"]
                        ),
                        ui.hr(),
                        ui.input_text("hf_repo", "HuggingFace Repository", value=MODEL_CONFIG["huggingface_repo"]),
                        ui.input_text("pin_name", "Vetiver Pin Name", placeholder="gradient_boosting"),
                        ui.hr(),
                        ui.input_action_button("load_model_btn", "Load Model from HuggingFace", class_="btn-admiral", style="width: 100%;"),
                        ui.hr(),
                        ui.output_ui("model_load_status")
                    )
                )
            ),
            ui.column(
                8,
                ui.card(
                    ui.card_header("Model Performance Metrics"),
                    ui.output_ui("model_performance_metrics")
                ),
                ui.card(
                    ui.card_header("Model Comparison"),
                    output_widget("model_comparison_plot")
                )
            )
        ),
        
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Model Retraining"),
                    ui.card_body(
                        ui.p("Upload new training data to retrain the model and improve predictions."),
                        ui.input_file("retrain_data_upload", "Upload Training Data (CSV)", accept=[".csv"]),
                        ui.hr(),
                        ui.row(
                            ui.column(
                                6,
                                ui.input_slider("train_test_split", "Train/Test Split", min=0.6, max=0.9, value=0.8, step=0.05)
                            ),
                            ui.column(
                                6,
                                ui.input_numeric("random_state", "Random State", value=42, min=1, max=1000)
                            )
                        ),
                        ui.hr(),
                        ui.input_action_button("retrain_model_btn", "Retrain Model", class_="btn-admiral"),
                        ui.hr(),
                        ui.output_ui("retraining_status")
                    )
                )
            )
        )
    ),
    
    # ========================================================================
    # TAB 5: EXPLAINABILITY & COMPLIANCE
    # ========================================================================
    ui.nav_panel(
        "📋 Explainability & Compliance",
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Model Explainability Dashboard"),
                    ui.p("Understanding model predictions for regulatory compliance and stakeholder transparency")
                )
            )
        ),
        
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Global Feature Importance"),
                    output_widget("global_shap_plot")
                )
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Feature Impact Summary"),
                    ui.output_ui("feature_impact_summary")
                )
            )
        ),
        
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("LIME Explanations (Select Claim)"),
                    ui.layout_sidebar(
                    ui.sidebar(
                        ui.input_select(
                            "selected_feature", 
                            "Select Feature for  Analysis", 
                            choices={},
                            selected = None
                        ),
                        title="Plot Controls",
                        # Adjust width as needed
                        width=250 
                    ),
                    # The main content area of the card
                    output_widget("lime_explanation_plot")
                ),
                full_screen = True,
            )
        ),
        
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Compliance Report"),
                    ui.output_ui("compliance_report"),
                    ui.hr(),
                    ui.download_button("download_compliance_report", "Download Compliance Report", class_="btn-admiral")
                )
            )
        )
    ),
    
    # ========================================================================
    # TAB 6: BUSINESS ANALYTICS
    # ========================================================================
    ui.nav_panel(
        "📈 Business Analytics",
        ui.row(
            ui.column(
                3,
                ui.card(
                    ui.card_header("Filter Options"),
                    ui.input_select(
                        "analytics_dimension",
                        "Analysis Dimension",
                        choices={
                            "claim_type": "By Claim Type",
                            "driver_age": "By Driver Age",
                            "vehicle_age": "By Vehicle Age",
                            "weather": "By Weather",
                            "severity": "By Severity"
                        }
                    ),
                    ui.input_slider("analytics_severity_threshold", "Severity Threshold (£)", 
                                  min=0, max=100000, value=15000, step=1000)
                )
            ),
            ui.column(
                9,
                ui.card(
                    ui.card_header("Key Performance Indicators"),
                    ui.row(
                        ui.column(3, ui.output_ui("kpi_prediction_accuracy")),
                        ui.column(3, ui.output_ui("kpi_processing_time")),
                        ui.column(3, ui.output_ui("kpi_cost_savings")),
                        ui.column(3, ui.output_ui("kpi_model_confidence"))
                    )
                )
            )
        ),
        
        ui.row(
            ui.column(
                6,
                ui.card(
                    ui.card_header("Claims Trend Analysis"),
                    output_widget("claims_trend_plot")
                )
            ),
            ui.column(
                6,
                ui.card(
                    ui.card_header("Reserve Adequacy Analysis"),
                    output_widget("reserve_adequacy_plot")
                )
            )
        ),
        
        ui.row(
            ui.column(
                12,
                ui.card(
                    ui.card_header("Strategic Insights & Recommendations"),
                    ui.output_ui("strategic_insights")
                )
            )
        )
    ),
    
    # Theme
    theme=shinyswatch.theme.flatly(),

    # Navigation bar configuration
    title=ui.tags.div(
        ui.img(src="Victor_Logo2.png", style="height: 30px; margin-right: 10px; vertical-align: middle;"),
        "FNOL Claims Intelligence",
        style="font-weight: 600;"
    ),
    id="main_navbar",
    footer=ui.div(
        {"style": "text-align: center; padding: 20px; background-color: #f8f9fa; margin-top: 40px;"},
        ui.p(f"© 2025 Admiral Group | Version {APP_CONFIG['version']} | Claims Analytics & Operations"),
        ui.tags.small("Powered by MLflow, Vetiver, Pins and Hugging Face")
    )
)