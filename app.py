# ============================================================
# EMPLOYEE ABSENTEEISM ANALYTICS DASHBOARD
# Portfolio release: five-tab dashboard, exports, and scenario reset
# ============================================================
# PURPOSE:
# Five-page interactive dashboard with linked exploration filters,
# statistical evidence, decision-support recommendations, and a
# hypothetical workforce coverage simulator with one-click reset.
# Exports respect selected filters; static evidence and recommendations
# deliberately use the full historical cohort.

# STEP 1: IMPORT LIBRARIES
from pathlib import Path
import calendar
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from dash import Dash, Input, Output, State, dcc, html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict

# STEP 2: DEFINE A CONSISTENT DASHBOARD THEME
NAVY = "#172B4D"
TEAL = "#0D9488"
PURPLE = "#6366F1"
GRAY = "#64748B"
BACKGROUND = "#F1F5F9"
WHITE = "#FFFFFF"

# STEP 3: LOAD THE PROJECT DATA
# The path is relative to app.py, so it works when the folder moves.
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / "Absenteeism_at_work.csv"
df = pd.read_csv(DATA_PATH, sep=";")
df.columns = df.columns.str.strip()  # Remove trailing column-name spaces.

# Positive-hour records are used for analyses of absence duration.
absence_data = df[df["Absenteeism time in hours"] > 0].copy()

# STEP 4: EXECUTIVE KPIs
total_records = len(df)
total_employees = df["ID"].nunique()
total_hours = df["Absenteeism time in hours"].sum()
positive_records = len(absence_data)

# STEP 5: PREPARE THE TOP REASONS CHART
reason_labels = {
    # Codes 1–21 follow the ICD chapter descriptions in the
    # project's Attribute Information document. Codes 22–28
    # are non-ICD absence categories in that same dictionary.
    1: "Infectious and Parasitic Diseases",
    2: "Neoplasms",
    3: "Blood and Immune Disorders",
    4: "Endocrine, Nutritional and Metabolic Diseases",
    5: "Mental and Behavioural Disorders",
    6: "Nervous System Diseases",
    7: "Eye and Adnexa Diseases",
    8: "Ear and Mastoid Diseases",
    9: "Circulatory System Diseases",
    10: "Respiratory Diseases",
    11: "Digestive Diseases",
    12: "Skin Conditions",
    13: "Musculoskeletal Conditions",
    14: "Genitourinary Diseases",
    15: "Pregnancy, Childbirth and Puerperium",
    16: "Perinatal Conditions",
    17: "Congenital and Chromosomal Conditions",
    18: "Symptoms and Abnormal Findings",
    19: "Injury and Poisoning",
    20: "External Causes of Morbidity and Mortality",
    21: "Health Status and Health Service Contact",
    22: "Patient Follow-up",
    23: "Medical Consultations",
    24: "Blood Donation",
    25: "Laboratory Examination",
    26: "Unjustified Absence",
    27: "Physiotherapy",
    28: "Dental Consultations",
}
reason_summary = (
    absence_data.groupby("Reason for absence")
    .agg(records=("Absenteeism time in hours", "size"),
         total_hours=("Absenteeism time in hours", "sum"))
    .reset_index()
)
top_reasons = reason_summary.nlargest(10, "total_hours").copy()
top_reasons["Reason"] = top_reasons["Reason for absence"].map(reason_labels)
top_reasons = top_reasons.sort_values("total_hours")

fig_reasons = px.bar(
    top_reasons, x="total_hours", y="Reason", orientation="h",
    title="Top 10 Reasons by Recorded Absence Hours",
    text="total_hours", custom_data=["records"],
    color_discrete_sequence=[TEAL],
)
fig_reasons.update_traces(
    textposition="outside",
    hovertemplate="<b>%{y}</b><br>Total Hours: %{x}"
                  "<br>Absence Records: %{customdata[0]}<extra></extra>",
)
fig_reasons.update_layout(
    height=550, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), xaxis_title="Total Recorded Absence Hours",
    yaxis_title="", margin=dict(l=20, r=35, t=70, b=50),
)

# STEP 6: PREPARE MONTHLY TRENDS
monthly_data = absence_data[absence_data["Month of absence"].between(1, 12)]
monthly_summary = (
    monthly_data.groupby("Month of absence")
    .agg(records=("Absenteeism time in hours", "size"),
         total_hours=("Absenteeism time in hours", "sum"))
    .reindex(range(1, 13), fill_value=0)
    .reset_index()
)
monthly_summary["Month"] = monthly_summary["Month of absence"].map(
    lambda month: calendar.month_name[month]
)
month_order = list(calendar.month_name)[1:]

fig_monthly_hours = px.bar(
    monthly_summary, x="Month", y="total_hours", text="total_hours",
    title="Recorded Absence Hours by Month",
    category_orders={"Month": month_order}, color_discrete_sequence=[TEAL],
)
fig_monthly_hours.update_traces(textposition="outside")
fig_monthly_hours.update_layout(
    height=440, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), xaxis_title="Month",
    yaxis_title="Recorded Absence Hours",
    margin=dict(l=30, r=20, t=75, b=90),
)
fig_monthly_hours.update_xaxes(tickangle=-45)

fig_monthly_records = px.line(
    monthly_summary, x="Month", y="records", markers=True,
    title="Number of Absence Records by Month",
    category_orders={"Month": month_order}, color_discrete_sequence=[PURPLE],
)
fig_monthly_records.update_traces(line=dict(width=3), marker=dict(size=9))
fig_monthly_records.update_layout(
    height=440, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), xaxis_title="Month",
    yaxis_title="Number of Absence Records",
    margin=dict(l=30, r=20, t=75, b=90),
)
fig_monthly_records.update_xaxes(tickangle=-45)

# STEP 7: PREPARE ABSENCE DURATION AND WEEKDAY PATTERNS
# The zoomed histogram preserves long absences in the underlying data.
fig_duration = px.histogram(
    df, x="Absenteeism time in hours",
    title="Distribution of Recorded Absence Durations",
    color_discrete_sequence=[PURPLE],
)
fig_duration.update_traces(xbins=dict(start=-0.5, end=120.5, size=1))
fig_duration.update_xaxes(range=[-0.5, 16.5], dtick=2)
fig_duration.update_layout(
    height=470, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), xaxis_title="Absence Duration (Hours)",
    yaxis_title="Number of Records",
    margin=dict(l=35, r=20, t=75, b=60),
)

weekday_labels = {2: "Monday", 3: "Tuesday", 4: "Wednesday",
                  5: "Thursday", 6: "Friday"}
weekday_data = absence_data[absence_data["Day of the week"].isin(weekday_labels)].copy()
weekday_data["extended"] = weekday_data["Absenteeism time in hours"] >= 16
weekday_data["extended_hours"] = weekday_data["Absenteeism time in hours"].where(
    weekday_data["extended"], 0
)
weekday_summary = (
    weekday_data.groupby("Day of the week")
    .agg(total_hours=("Absenteeism time in hours", "sum"),
         extended_hours=("extended_hours", "sum"))
    .reindex([2, 3, 4, 5, 6], fill_value=0).reset_index()
)
weekday_summary["Weekday"] = weekday_summary["Day of the week"].map(weekday_labels)
weekday_summary["shorter_hours"] = (
    weekday_summary["total_hours"] - weekday_summary["extended_hours"]
)
fig_weekday = go.Figure()
fig_weekday.add_trace(go.Bar(
    x=weekday_summary["Weekday"], y=weekday_summary["shorter_hours"],
    name="Shorter (<16 Hours)", marker_color=PURPLE,
    text=weekday_summary["shorter_hours"], textposition="inside",
))
fig_weekday.add_trace(go.Bar(
    x=weekday_summary["Weekday"], y=weekday_summary["extended_hours"],
    name="Extended (16+ Hours)", marker_color=TEAL,
    text=weekday_summary["extended_hours"], textposition="inside",
))
fig_weekday.update_layout(
    title="Shorter vs. Extended Absences by Weekday", barmode="stack",
    height=470, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), xaxis_title="Weekday",
    yaxis_title="Recorded Absence Hours", legend_title_text="Absence Duration",
    legend=dict(orientation="h", y=-0.25, x=0),
    margin=dict(l=40, r=20, t=75, b=100),
)

# STEP 8: PREPARE SEASONAL PATTERNS
# Season names are inferred from the month-to-season cross-tab.
season_labels = {1: "Summer", 2: "Winter", 3: "Spring", 4: "Fall"}
season_data = absence_data[absence_data["Seasons"].isin(season_labels)]
season_summary = (
    season_data.groupby("Seasons")
    .agg(records=("Absenteeism time in hours", "size"),
         total_hours=("Absenteeism time in hours", "sum"))
    .reset_index()
)
season_summary["Season"] = season_summary["Seasons"].map(season_labels)
season_summary["Record Share (%)"] = (
    season_summary["records"] / season_summary["records"].sum() * 100
)
season_summary["Hour Share (%)"] = (
    season_summary["total_hours"] / season_summary["total_hours"].sum() * 100
)
season_long = season_summary.melt(
    id_vars="Season", value_vars=["Record Share (%)", "Hour Share (%)"],
    var_name="Metric", value_name="Percentage",
)
fig_seasons = px.bar(
    season_long, x="Season", y="Percentage", color="Metric",
    barmode="group", title="Seasonal Absence Frequency vs. Total Hours",
    category_orders={"Season": ["Winter", "Spring", "Summer", "Fall"]},
    color_discrete_map={"Record Share (%)": PURPLE, "Hour Share (%)": TEAL},
    text="Percentage",
)
fig_seasons.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_seasons.update_layout(
    height=460, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), xaxis_title="Season",
    yaxis_title="Share of Total (%)", yaxis_range=[0, 35],
    legend_title_text="Measurement", margin=dict(l=50, r=30, t=80, b=55),
)

# Calculate summary metrics displayed in the Patterns tab.
extended_hours_total = absence_data.loc[
    absence_data["Absenteeism time in hours"] >= 16,
    "Absenteeism time in hours",
].sum()
extended_share = 100 * extended_hours_total / total_hours
early_week_hours = weekday_data.loc[
    weekday_data["Day of the week"].isin([2, 3, 4]),
    "Absenteeism time in hours",
].sum()
early_week_share = 100 * early_week_hours / total_hours
spring_hours = season_summary.loc[
    season_summary["Season"] == "Spring", "total_hours"
].sum()
spring_share = 100 * spring_hours / total_hours

# ============================================================
# NEW IN STEP 18H: FACTORS AND STATISTICAL EVIDENCE
# ============================================================

# STEP 9: PREPARE EMPLOYEE-LEVEL STATISTICAL DATA
# Each row represents a distinct employee rather than an
# absence record. This helps avoid treating repeated records
# from the same person as independent employees.
employee_summary = (
    absence_data.groupby("ID")
    .agg(
        commute_km=("Distance from Residence to Work", "median"),
        service_time=("Service time", "median"),
        average_hours=("Absenteeism time in hours", "mean"),
        absence_records=("Absenteeism time in hours", "size"),
    )
    .reset_index()
)
# Match the two-decimal employee averages used in the notebook.
employee_summary["average_hours"] = employee_summary["average_hours"].round(2)
employee_age = df.groupby("ID")["Age"].median().rename("age").reset_index()
factor_data = employee_summary.merge(employee_age, on="ID", how="inner")

# STEP 10: EMPLOYEE-LEVEL SPEARMAN CORRELATION HEATMAP
# Spearman correlation describes ranked associations.
# A correlation is not evidence of a causal relationship.
factor_columns = ["commute_km", "service_time", "age", "average_hours"]
correlation_matrix = factor_data[factor_columns].corr(method="spearman")
correlation_matrix = correlation_matrix.rename(
    index={"commute_km": "Commute", "service_time": "Service Time",
           "age": "Age", "average_hours": "Avg. Duration"},
    columns={"commute_km": "Commute", "service_time": "Service Time",
             "age": "Age", "average_hours": "Avg. Duration"},
)
fig_correlations = px.imshow(
    correlation_matrix,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1,
    title="Employee-Level Correlations (Spearman)",
)
fig_correlations.update_layout(
    height=480, paper_bgcolor=WHITE,
    font=dict(color=NAVY), margin=dict(l=40, r=30, t=80, b=40),
)

# STEP 11: WORKLOAD-LEVEL SCATTER PLOT
# Shared workload measurements are grouped into 37 distinct
# values, rather than treated as independent employee workloads.
workload_summary = (
    absence_data.groupby("Work load Average/day")
    .agg(
        records=("Absenteeism time in hours", "size"),
        average_hours=("Absenteeism time in hours", "mean"),
        employees=("ID", "nunique"),
    )
    .reset_index()
)
fig_workload = px.scatter(
    workload_summary,
    x="Work load Average/day", y="average_hours",
    size="records", size_max=30,
    hover_data=["employees", "records"],
    title="Workload vs. Average Recorded Absence Duration",
    labels={"Work load Average/day": "Recorded Workload Average/Day",
            "average_hours": "Average Absence Duration (Hours)"},
    color_discrete_sequence=[TEAL],
)
fig_workload.update_layout(
    height=480, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), margin=dict(l=40, r=20, t=80, b=55),
)

# STEP 12: REGRESSION AND OUT-OF-SAMPLE VALIDATION
# We fit the same exploratory three-predictor model used in
# the notebook, with log1p-transformed average duration.
X = factor_data[["commute_km", "service_time", "age"]]
y = np.log1p(factor_data["average_hours"])
regression = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")

# Leave-one-employee-out validation tests each person using
# a model trained without that person's observation.
loo = LeaveOneOut()
model_preds = cross_val_predict(LinearRegression(), X, y, cv=loo)
baseline_preds = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=loo)
model_mae = mean_absolute_error(y, model_preds)
baseline_mae = mean_absolute_error(y, baseline_preds)
model_cv_r2 = r2_score(y, model_preds)
baseline_cv_r2 = r2_score(y, baseline_preds)

# This figure compares the absolute prediction errors.
# Lower MAE is better. MAE is measured on the log scale.
validation_results = pd.DataFrame({
    "Model": ["Multiple Regression", "Mean Baseline"],
    "MAE": [model_mae, baseline_mae],
})
fig_validation = px.bar(
    validation_results, x="Model", y="MAE", text="MAE",
    title="Leave-One-Employee-Out Prediction Error",
    color="Model",
    color_discrete_map={"Multiple Regression": TEAL,
                        "Mean Baseline": PURPLE},
)
fig_validation.update_traces(texttemplate="%{text:.4f}", textposition="outside")
fig_validation.update_layout(
    height=430, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
    font=dict(color=NAVY), yaxis_title="MAE (log scale)",
    xaxis_title="", showlegend=False,
    margin=dict(l=45, r=25, t=80, b=60),
)

# ============================================================
# NEW IN STEP 18I: EVIDENCE-BASED BUSINESS RECOMMENDATIONS
# ============================================================

# STEP 12B: CALCULATE RECOMMENDATION EVIDENCE
# Recalculate the supporting metrics directly from the CSV.
# This makes the recommendations auditable and keeps the
# dashboard independent of calculations made in Jupyter.

# The 16-hour threshold is an analytical choice, not an
# official definition of an extended absence.
extended_absences = absence_data[
    absence_data["Absenteeism time in hours"] >= 16
].copy()
extended_records = len(extended_absences)
extended_record_share = 100 * extended_records / positive_records

# Codes 23 and 28 denote medical and dental consultations.
# The combined record share describes frequency, not risk.
consultation_records = absence_data[
    absence_data["Reason for absence"].isin([23, 28])
]
consultation_record_share = 100 * len(consultation_records) / positive_records

# Identify the two months with the largest recorded hours.
# The CSV does not contain a year column, so these are pooled
# historical totals rather than confirmed recurring peaks.
peak_months = monthly_summary.nlargest(2, "total_hours")
peak_month_names = " and ".join(peak_months["Month"])
peak_month_hours = int(peak_months["total_hours"].sum())
peak_month_share = 100 * peak_month_hours / total_hours

# Summarize extended hours in the two largest categories.
# These data are aggregate only; no personal medical records
# or individual employee scores will appear on this page.
long_reason_hours = extended_absences.loc[
    extended_absences["Reason for absence"].isin([13, 19]),
    "Absenteeism time in hours",
].sum()
long_reason_share = 100 * long_reason_hours / extended_hours_total

# STEP 12C: LINK EACH FINDING TO AN ACTION AND A VALIDATION PLAN
# Priorities below are *proposed* business priorities based on
# the descriptive evidence; none is a proven intervention.
recommendation_rows = [
    {
        "Priority": "High",
        "Focus": "Extended absence coverage",
        "Evidence": (
            f"{extended_records} extended records ({extended_record_share:.1f}% "
            f"of positive-hour records) account for "
            f"{int(extended_hours_total):,} hours "
            f"({extended_share:.1f}% of recorded hours)."
        ),
        "Proposed action": (
            "Cross-train essential roles and design voluntary, "
            "policy-compliant backup coverage arrangements for prolonged leave."
        ),
        "What to validate": (
            "Collect staffing rosters and track unfilled coverage hours, "
            "backup response times, and overtime before and after a pilot."
        ),
    },
    {
        "Priority": "High",
        "Focus": "Planned appointment flexibility",
        "Evidence": (
            f"Medical and dental consultations represent "
            f"{len(consultation_records)} of {positive_records} positive-hour "
            f"records ({consultation_record_share:.1f}%)."
        ),
        "Proposed action": (
            "Evaluate optional scheduling flexibility for planned appointments "
            "while respecting employees' health privacy and leave rights."
        ),
        "What to validate": (
            "Track staffing coverage gaps and employee feedback; "
            "confirm whether flexibility is feasible for each role."
        ),
    },
    {
        "Priority": "Investigate",
        "Focus": "Early-week staffing patterns",
        "Evidence": (
            f"Monday through Wednesday account for "
            f"{early_week_share:.1f}% of recorded absence hours."
        ),
        "Proposed action": (
            "Review schedules and coverage demand before adjusting "
            "staffing levels on any weekday."
        ),
        "What to validate": (
            "Obtain scheduled hours by weekday and determine how "
            "multi-day absences were assigned to a single weekday code."
        ),
    },
    {
        "Priority": "Investigate",
        "Focus": "Monthly workforce planning",
        "Evidence": (
            f"{peak_month_names} contribute {peak_month_hours:,} hours "
            f"({peak_month_share:.1f}% of recorded hours)."
        ),
        "Proposed action": (
            "Investigate reasons and extended records in higher-hour "
            "months before changing recurring staff schedules."
        ),
        "What to validate": (
            "Collect actual dates, year labels, workforce size, and "
            "scheduled hours to determine whether peaks recur."
        ),
    },
]

# Store recommendations in a DataFrame so the aggregated
# evidence can be downloaded as a stakeholder-friendly CSV.
recommendations_df = pd.DataFrame(recommendation_rows)

# ============================================================
# DASH APPLICATION, REUSABLE COMPONENTS, AND PAGES
# ============================================================

# STEP 13: START THE DASH APP
app = Dash(__name__)
app.title = "Employee Absenteeism Analytics"

# Keep the simulator values fully readable on browsers that display
# oversized native +/- controls inside numeric inputs. CSS is embedded
# in app.py so no separate stylesheet is required when replacing the file.
SCENARIO_INPUT_CSS = """
<style>
input.scenario-number-input {
    appearance: textfield !important;
    -moz-appearance: textfield !important;
    display: block;
    width: 100% !important;
    min-width: 0 !important;
    height: 44px !important;
    line-height: 24px !important;
    padding: 8px 12px !important;
    box-sizing: border-box !important;
    font-variant-numeric: tabular-nums;
    background-color: #FFFFFF;
}
input.scenario-number-input::-webkit-outer-spin-button,
input.scenario-number-input::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    display: none !important;
    margin: 0 !important;
}
</style>
"""
app.index_string = app.index_string.replace(
    "{%css%}", "{%css%}" + SCENARIO_INPUT_CSS
)

# A production WSGI process can serve this app with: gunicorn app:server
server = app.server

# Reusable metric cards make dashboard styling consistent.
def kpi_card(title, value, description):
    return html.Div([
        html.P(title, style={"color": GRAY, "fontSize": "14px", "margin": "0"}),
        html.H2(value, style={"color": NAVY, "fontSize": "32px", "margin": "12px 0"}),
        html.P(description, style={"color": GRAY, "fontSize": "12px", "margin": "0"}),
    ], style={
        "backgroundColor": WHITE, "padding": "24px", "borderRadius": "12px",
        "border": "1px solid #E2E8F0", "flex": "1 1 210px",
        "boxShadow": "0 2px 6px rgba(0,0,0,0.04)",
    })

# Chart cards contain interactive figures in white panels.
def chart_card(figure, graph_id=None):
    # Dash permits omitting a Graph ID, but does not permit id=None.
    # Add an ID only for charts controlled by a callback.
    graph_props = {
        "figure": figure,
        "config": {"displayModeBar": True},
        "style": {"width": "100%"},
    }
    if graph_id is not None:
        graph_props["id"] = graph_id

    return html.Div([
        dcc.Graph(**graph_props)
    ], style={
        "backgroundColor": WHITE, "borderRadius": "12px",
        "border": "1px solid #E2E8F0", "padding": "12px",
        "minWidth": "0", "flex": "1 1 480px",
    })

# Reuse the same layout for side-by-side chart rows.
def chart_row(*figures):
    return html.Div([
        chart_card(item[0], item[1]) if isinstance(item, tuple)
        else chart_card(item) for item in figures
    ], style={
        "display": "flex", "gap": "20px", "flexWrap": "wrap"
    })

# Reusable small-text note for methodological caveats.
def note(text):
    return html.P(text, style={
        "color": GRAY, "fontSize": "13px", "marginTop": "16px",
        "lineHeight": "1.6",
    })

# Navigation styles, including a teal underline for the active tab.
tab_style = {
    "padding": "16px 10px", "fontWeight": "600", "color": GRAY,
    "backgroundColor": WHITE, "borderTop": "none",
    "borderBottom": "2px solid #E2E8F0",
}
selected_tab_style = {
    **tab_style, "color": TEAL,
    "borderBottom": f"3px solid {TEAL}",
    "backgroundColor": "#F8FAFC",
}

# STEP 14: EXECUTIVE OVERVIEW -- PRESERVED
# This page keeps the existing KPI cards and three charts.
overview_content = html.Div([
    html.H2("Executive Overview", style={"color": NAVY}),
    # The linked filters below replace these KPI cards dynamically.
    html.Div(id="overview-filtered-kpis", children=[
        kpi_card("Total Records", f"{total_records:,}", "All dataset observations"),
        kpi_card("Unique Employees", f"{total_employees:,}", "Distinct employee IDs"),
        kpi_card("Recorded Absence Hours", f"{total_hours:,}",
                 "Total hours across all records"),
        kpi_card("Positive-Hour Records", f"{positive_records:,}",
                 "Records with more than zero absence hours"),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
              "marginTop": "25px", "marginBottom": "30px"}),
    chart_card(fig_reasons, "overview-reasons-chart"),
    html.H2("Monthly Absenteeism Trends", style={
        "color": NAVY, "marginTop": "40px", "marginBottom": "20px"
    }),
    chart_row(
        (fig_monthly_hours, "overview-monthly-hours-chart"),
        (fig_monthly_records, "overview-monthly-records-chart")
    ),
    note("Charts and KPIs follow the filters above. Monthly totals aggregate "
         "records across multiple years; they do not establish recurring seasonal trends."),
    note("Recorded hours do not represent an absenteeism rate because "
         "scheduled working hours are unavailable."),
    # STEP 21A: PRACTICAL EXPORTS FOR THE ACTIVE FILTERS
    html.H2("Export the Analysis", style={
        "color": NAVY, "marginTop": "38px", "marginBottom": "12px"
    }),
    html.Div([
        html.P(
            "Export the current selection. The detailed CSV omits employee IDs "
            "and personal profile columns. The ZIP contains additional "
            "Excel-readable CSV reports for months, reasons, weekdays, seasons, "
            "durations, and management recommendations.",
            style={"color": GRAY, "margin": "0 0 15px", "lineHeight": "1.6"}
        ),
        html.Div([
            html.Button(
                "Download Filtered Records (CSV)",
                id="download-filtered-records-button", n_clicks=0,
                style={"backgroundColor": NAVY, "color": WHITE,
                       "padding": "12px 16px", "border": "none",
                       "borderRadius": "8px", "fontWeight": "600",
                       "cursor": "pointer"}
            ),
            html.Button(
                "Download Full Analysis Pack (ZIP)",
                id="download-analysis-pack-button", n_clicks=0,
                style={"backgroundColor": TEAL, "color": WHITE,
                       "padding": "12px 16px", "border": "none",
                       "borderRadius": "8px", "fontWeight": "600",
                       "cursor": "pointer"}
            ),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        html.P(
            "Exports are historical descriptions, not absenteeism rates. "
            "Removing employee IDs does not guarantee full anonymity; "
            "review data-sharing obligations before publishing record-level exports. "
            "Business recommendations in the ZIP refer to the full cohort.",
            style={"color": GRAY, "fontSize": "12px", "lineHeight": "1.6",
                   "margin": "14px 0 0"}
        ),
        dcc.Download(id="filtered-records-download"),
        dcc.Download(id="analysis-pack-download"),
    ], style={"backgroundColor": WHITE, "border": "1px solid #E2E8F0",
              "borderRadius": "12px", "padding": "20px"}),
], style={"padding": "25px 0"})

# STEP 15: ABSENTEEISM PATTERNS -- PRESERVED
patterns_content = html.Div([
    html.H2("Absenteeism Patterns", style={"color": NAVY}),
    html.P("Explore how absence duration and recorded hours vary "
           "across weekdays and seasons.", style={"color": GRAY}),
    html.Div(id="patterns-filtered-kpis", children=[
        kpi_card("Extended Absence Hours", f"{extended_share:.1f}%",
                 "Share of hours from 16+ hour records"),
        kpi_card("Monday–Wednesday Hours", f"{early_week_share:.1f}%",
                 "Share attributed to early-week records"),
        kpi_card("Spring Absence Hours", f"{spring_share:.1f}%",
                 "Share of total recorded absence hours"),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
              "marginBottom": "30px"}),
    chart_row(
        (fig_duration, "patterns-duration-chart"),
        (fig_weekday, "patterns-weekday-chart")
    ),
    note("The duration histogram usually focuses on 0–16 hours and expands "
         "for an extended-only selection. Longer records remain in the "
         "dataset. Extended hours may span multiple days but are associated "
         "with one weekday code."),
    html.H2("Seasonal Absenteeism", style={
        "color": NAVY, "marginTop": "35px", "marginBottom": "20px"
    }),
    chart_card(fig_seasons, "patterns-seasons-chart"),
    note("Shares are recalculated within the selected subset, so filtering "
         "to one season yields 100% for that season. Season labels are inferred "
         "from month-season coding; these are not absenteeism rates."),
], style={"padding": "25px 0"})

# STEP 16: FACTORS & EVIDENCE -- NEW
# Report findings and limitations; do not identify individual
# employees or use age for individual employment decisions.
factors_content = html.Div([
    html.H2("Factors & Statistical Evidence", style={"color": NAVY}),
    html.P("Full historical cohort (not affected by exploration filters): "
           "employee-level associations, workload patterns, "
           "and the reliability of our preliminary regression model.",
           style={"color": GRAY}),
    html.Div([
        kpi_card("Employees Analyzed", f"{len(factor_data):,}",
                 "Employees with positive-hour absence records"),
        kpi_card("In-Sample R²", f"{regression.rsquared:.1%}",
                 "Explained variation in log average duration"),
        kpi_card("Cross-Validated R²", f"{model_cv_r2:.3f}",
                 "Regression performance on held-out employees"),
        kpi_card("Baseline CV R²", f"{baseline_cv_r2:.3f}",
                 "Training-mean prediction baseline"),
    ], style={"display": "flex", "gap": "20px", "flexWrap": "wrap",
              "marginTop": "25px", "marginBottom": "30px"}),
    chart_row(fig_correlations, fig_workload),
    note("Correlations are exploratory associations, not causal effects. "
         "Workload values may be shared across reporting periods, "
         "so records are not independent workload measurements."),
    html.H2("Predictive Validation", style={
        "color": NAVY, "marginTop": "35px", "marginBottom": "20px"
    }),
    chart_card(fig_validation),
    html.Div([
        html.H3("Interpretation", style={"color": NAVY, "marginTop": "0"}),
        html.P(
            f"Multiple regression LOOCV MAE: {model_mae:.4f}. "
            f"Mean baseline LOOCV MAE: {baseline_mae:.4f}. "
            "Lower error is better. The regression does not outperform "
            "the baseline on this dataset.",
            style={"color": GRAY, "lineHeight": "1.7"},
        ),
        html.P(
            "The model is exploratory and should not be used to predict "
            "or make decisions about individual employees' attendance.",
            style={"color": GRAY, "lineHeight": "1.7"},
        ),
    ], style={
        "backgroundColor": WHITE, "borderRadius": "12px", "padding": "22px",
        "border": "1px solid #E2E8F0", "marginTop": "20px",
    }),
    note("Validation uses 33 employees and a log-transformed target. "
         "Neither these metrics nor in-sample R² represent prediction accuracy "
         "in hours. The model's usefulness is limited by the small sample."),
], style={"padding": "25px 0"})

# ============================================================
# STEP 20A: WORKFORCE COVERAGE SCENARIO HELPERS
# ============================================================
# This is a hypothetical planning calculator. Its inputs are
# supplied by the dashboard user and are NOT predictions from
# historical absenteeism records or actual staffing schedules.

def scenario_number_input(label, component_id, default, description, *, max_value=None):
    """Build one editable numeric scenario assumption with guidance."""
    return html.Div([
        html.Label(label, htmlFor=component_id, style={
            "display": "block", "fontSize": "13px", "fontWeight": "600",
            "color": NAVY, "marginBottom": "7px",
        }),
        dcc.Input(
            id=component_id, className="scenario-number-input",
            type="number", value=default, min=0,
            step=1, debounce=False,
            **({"max": max_value} if max_value is not None else {}),
            style={"width": "100%", "boxSizing": "border-box",
                   "height": "44px", "lineHeight": "24px",
                   "border": "1px solid #CBD5E1", "borderRadius": "7px",
                   "padding": "8px 12px", "color": NAVY, "fontSize": "16px",
                   "backgroundColor": WHITE},
        ),
        html.P(description, style={"color": GRAY, "fontSize": "11px",
                                   "lineHeight": "1.5", "marginBottom": "0"}),
    ])


def compute_coverage_scenario(staff, shift_hours, unavailable,
                              needed_hours, backup_staff, backup_hours,
                              availability_pct):
    """Return inputs and outputs for a SINGLE hypothetical planning period.

    Regular usable hours = (scheduled staff - unavailable staff) * shift hours
    Backup usable hours = backup staff * backup hours * availability proportion
    Coverage gap = max(required coverage hours - usable hours, 0)
    """
    values = [staff, shift_hours, unavailable, needed_hours,
              backup_staff, backup_hours, availability_pct]
    if any(value is None for value in values):
        raise ValueError("Enter a number in every scenario field.")
    try:
        numeric = [float(value) for value in values]
    except (TypeError, ValueError):
        raise ValueError("All scenario inputs must be numbers.") from None
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError("Use finite numeric values in every field.")
    if any(value < 0 for value in numeric):
        raise ValueError("Scenario values cannot be negative.")
    staff, shift_hours, unavailable, needed_hours, backup_staff, backup_hours, availability_pct = numeric
    if any(not value.is_integer() for value in (staff, unavailable, backup_staff)):
        raise ValueError("Enter whole numbers for staff counts.")
    if unavailable > staff:
        raise ValueError("Unavailable staff cannot exceed scheduled staff.")
    if shift_hours > 24 or backup_hours > 24:
        raise ValueError("Hours per person cannot exceed 24 in one daily planning period.")
    if availability_pct > 100:
        raise ValueError("Backup availability must be between 0% and 100%.")

    planned_hours = staff * shift_hours
    absence_hours = unavailable * shift_hours
    remaining_hours = planned_hours - absence_hours
    backup_capacity = backup_staff * backup_hours * availability_pct / 100
    covered_hours = remaining_hours + backup_capacity
    gap_before = max(0.0, needed_hours - remaining_hours)
    gap_after = max(0.0, needed_hours - covered_hours)
    gap_recovered = gap_before - gap_after
    covered_share = min(100, covered_hours / needed_hours * 100) if needed_hours > 0 else 100.0

    return {
        "Scheduled staff": int(staff),
        "Hours per scheduled staff member": shift_hours,
        "Unavailable scheduled staff": int(unavailable),
        "Required coverage hours": needed_hours,
        "Backup staff": int(backup_staff),
        "Hours per backup staff member": backup_hours,
        "Backup availability (%)": availability_pct,
        "Originally scheduled staff hours": planned_hours,
        "Assumed unavailable staff hours": absence_hours,
        "Available hours without backup": remaining_hours,
        "Effective backup hours": backup_capacity,
        "Available hours with backup": covered_hours,
        "Coverage gap without backup": gap_before,
        "Coverage gap with backup": gap_after,
        "Coverage gap reduced by": gap_recovered,
        "Covered share of required hours (%)": covered_share,
        "Unused available hours": max(0, covered_hours - needed_hours),
    }


def coverage_comparison_figure(results):
    """Compare hypothetical available capacity against required hours."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Without backup", "With backup"],
        y=[results["Available hours without backup"],
           results["Available hours with backup"]],
        marker_color=[PURPLE, TEAL],
        text=[f"{results['Available hours without backup']:,.1f} h",
              f"{results['Available hours with backup']:,.1f} h"],
        textposition="outside",
        hovertemplate="%{x}: %{y:.1f} usable hours<extra></extra>",
    ))
    fig.add_hline(
        y=results["Required coverage hours"], line_dash="dash",
        line_color=NAVY,
        annotation_text=f"Required: {results['Required coverage hours']:,.1f} h",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Hypothetical Usable Coverage Hours",
        yaxis_title="Staff coverage hours in selected period",
        xaxis_title="", showlegend=False,
        yaxis_range=[0, max(8.0, results["Required coverage hours"],
                            results["Available hours with backup"]) * 1.25],
        height=440, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(color=NAVY),
        margin=dict(l=45, r=25, t=85, b=55),
    )
    return fig


# STEP 17: BUSINESS RECOMMENDATIONS -- COMPLETE
# Provide a practical decision framework, a priority filter,
# proposed success measures, and the evidence limitations.
# The interactive cards are populated by a Dash callback below.

def recommendation_card(item):
    """Return a management-friendly card for one proposed action."""
    is_high = item["Priority"] == "High"
    accent = TEAL if is_high else PURPLE
    return html.Div([
        html.Div([
            html.Span(
                f"Proposed priority: {item['Priority']}",
                style={"fontSize": "12px", "fontWeight": "700", "color": accent},
            ),
        ], style={"marginBottom": "10px"}),
        html.H3(item["Focus"], style={
            "color": NAVY, "margin": "0 0 14px", "fontSize": "19px"
        }),
        html.P("Observed evidence", style={
            "fontSize": "12px", "color": GRAY, "fontWeight": "700",
            "marginBottom": "4px"
        }),
        html.P(item["Evidence"], style={
            "color": NAVY, "lineHeight": "1.55", "marginTop": "0"
        }),
        html.P("Potential management action", style={
            "fontSize": "12px", "color": GRAY, "fontWeight": "700",
            "marginBottom": "4px"
        }),
        html.P(item["Proposed action"], style={
            "color": NAVY, "lineHeight": "1.55", "marginTop": "0"
        }),
        html.P("Evidence needed before adoption", style={
            "fontSize": "12px", "color": GRAY, "fontWeight": "700",
            "marginBottom": "4px"
        }),
        html.P(item["What to validate"], style={
            "color": GRAY, "lineHeight": "1.55", "marginTop": "0"
        }),
    ], style={
        "backgroundColor": WHITE, "borderRadius": "12px",
        "border": "1px solid #E2E8F0", "borderLeft": f"4px solid {accent}",
        "padding": "22px", "flex": "1 1 440px", "minWidth": "0",
    })

# A compact, reusable row for proposed operational measures.
def measurement_row(name, purpose, data_needed):
    return html.Tr([
        html.Td(name, style={"padding": "13px", "fontWeight": "600"}),
        html.Td(purpose, style={"padding": "13px"}),
        html.Td(data_needed, style={"padding": "13px"}),
    ], style={"borderBottom": "1px solid #E2E8F0"})

recommendations_content = html.Div([
    html.H2("Business Recommendations", style={"color": NAVY}),
    html.P(
        "Full historical cohort (not affected by exploration filters). "
        "Turn findings into testable workforce-planning actions. "
        "Proposed priorities are not proven methods of reducing absence.",
        style={"color": GRAY, "lineHeight": "1.7"}
    ),
    html.Div([
        kpi_card("Hours from extended records", f"{extended_share:.1f}%",
                 "16+ hour records; selected analytical threshold"),
        kpi_card("Appointment-related records", f"{consultation_record_share:.1f}%",
                 "Medical and dental consultations"),
        kpi_card("Early-week recorded hours", f"{early_week_share:.1f}%",
                 "Attributed to Monday–Wednesday records"),
    ], style={
        "display": "flex", "gap": "20px", "flexWrap": "wrap",
        "marginTop": "24px", "marginBottom": "28px",
    }),
    html.Div([
        html.Div([
            html.H3("Decision Priorities", style={
                "color": NAVY, "marginBottom": "6px"
            }),
            html.P("Filter the recommendations by proposed priority.",
                   style={"color": GRAY, "fontSize": "13px"}),
        ]),
        html.Div([
            html.Label("Show", htmlFor="recommendation-priority-filter",
                       style={"fontSize": "13px", "color": GRAY}),
            dcc.Dropdown(
                id="recommendation-priority-filter",
                options=[
                    {"label": "All priorities", "value": "All"},
                    {"label": "High", "value": "High"},
                    {"label": "Investigate", "value": "Investigate"},
                ],
                value="All", clearable=False,
                style={"minWidth": "190px"},
            ),
        ], style={"minWidth": "190px", "width": "220px"}),
    ], style={
        "display": "flex", "alignItems": "end", "justifyContent": "space-between",
        "gap": "20px", "flexWrap": "wrap", "marginBottom": "20px"
    }),
    # This container is updated in response to the dropdown.
    html.Div(id="recommendation-card-grid", style={
        "display": "flex", "gap": "18px", "flexWrap": "wrap"
    }),
    html.Div([
        html.Button(
            "Download Evidence & Recommendations (CSV)",
            id="download-recommendations-button",
            n_clicks=0,
            style={
                "backgroundColor": NAVY, "color": WHITE, "border": "none",
                "padding": "13px 18px", "borderRadius": "8px",
                "cursor": "pointer", "fontWeight": "600", "fontSize": "14px"
            },
        ),
        dcc.Download(id="recommendations-download"),
    ], style={"marginTop": "22px"}),
    # The simulator has its own dedicated navigation tab.
    html.P(
        "To explore hypothetical staffing capacity and backup coverage, "
        "open the Workforce Simulator tab. Its calculations use scenario "
        "assumptions rather than historical staff schedules.",
        style={"color": GRAY, "fontSize": "14px", "marginTop": "20px"},
    ),

    html.H2("How Management Could Measure Results", style={
        "color": NAVY, "marginTop": "40px", "marginBottom": "12px"
    }),
    html.P(
        "These are proposed operational KPIs for a future pilot; "
        "they cannot be calculated from the current dataset.",
        style={"color": GRAY, "lineHeight": "1.6"}
    ),
    html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th("Potential KPI", style={"padding": "13px"}),
                html.Th("Decision supported", style={"padding": "13px"}),
                html.Th("Additional data required", style={"padding": "13px"}),
            ], style={"backgroundColor": "#F8FAFC", "textAlign": "left"})),
            html.Tbody([
                measurement_row(
                    "Unfilled coverage hours",
                    "Assess whether contingency staffing reduces uncovered shifts.",
                    "Scheduled shifts and actual staffing coverage"
                ),
                measurement_row(
                    "Coverage response time",
                    "Evaluate how quickly backup staff can fill a gap.",
                    "Shift-change and replacement timestamps"
                ),
                measurement_row(
                    "Scheduling conflict count",
                    "Evaluate whether optional flexibility reduces disruption.",
                    "Scheduling requests and conflicts, in aggregate"
                ),
                measurement_row(
                    "Absence hours / scheduled hours",
                    "Establish a properly defined absenteeism rate.",
                    "Scheduled working hours for each reporting period"
                ),
            ]),
        ], style={
            "width": "100%", "borderCollapse": "collapse",
            "color": NAVY, "fontSize": "13px"
        }),
    ], style={
        "backgroundColor": WHITE, "borderRadius": "12px",
        "border": "1px solid #E2E8F0", "padding": "16px",
        "overflowX": "auto",
    }),
    html.H2("Decision Safeguards & Dataset Limitations", style={
        "color": NAVY, "marginTop": "40px", "marginBottom": "12px"
    }),
    html.Div([
        html.P(
            f"The dataset contains {total_records:,} records across "
            f"{total_employees} employees, but no scheduled-hour denominators "
            "or exact absence dates. Weekday and month totals cannot be "
            "interpreted as absence rates or proven recurring trends.",
            style={"color": NAVY, "lineHeight": "1.7"}
        ),
        html.P(
            f"Extended absence means 16+ hours for this analysis. "
            f"Musculoskeletal conditions and injury/poisoning contribute "
            f"{int(long_reason_hours):,} hours ({long_reason_share:.1f}% of "
            "extended hours), but recorded diagnoses do not establish "
            "work-related causes.",
            style={"color": NAVY, "lineHeight": "1.7"}
        ),
        html.P(
            "The preliminary employee-level regression performed worse "
            "than a mean-prediction baseline on held-out employees. "
            "Do not use it for individual attendance predictions, "
            "disciplinary decisions, or employment actions. Protect medical "
            "privacy and assess any intervention for fairness and feasibility.",
            style={"color": NAVY, "lineHeight": "1.7", "marginBottom": "0"}
        ),
    ], style={
        "backgroundColor": WHITE, "borderRadius": "12px",
        "border": "1px solid #E2E8F0", "padding": "24px",
    }),
], style={"padding": "25px 0"})

# STEP 18: WORKFORCE SIMULATOR -- DEDICATED FIFTH TAB
# Move the existing validated scenario inputs and outputs into
# their own page; callback IDs and calculations remain unchanged.
simulator_content = html.Div([
    # ========================================================
    # STEP 20B: INTERACTIVE COVERAGE SCENARIO SIMULATOR
    # ========================================================
    # All numbers below are editable scenario assumptions.
    # No staffing schedule or attendance risk is inferred from
    # the historical data (which lacks scheduled working hours).
    html.H2("Workforce Coverage Scenario Simulator", style={
        "color": NAVY, "marginTop": "0", "marginBottom": "9px",
    }),
    html.P(
        "Explore a hypothetical single-shift coverage plan. Change staffing, "
        "assumed unavailability, required hours, and backup availability "
        "to see estimated capacity gaps update immediately.",
        style={"color": GRAY, "lineHeight": "1.7"},
    ),
    html.Div([
        # Restore all hypothetical inputs with one click. This button
        # only affects the Workforce Simulator, not historical data filters.
        html.Div([
            html.Button(
                "Reset Scenario", id="sim-reset-button", n_clicks=0,
                title="Restore the original workforce planning assumptions",
                style={
                    "backgroundColor": WHITE, "color": NAVY,
                    "border": "1px solid #CBD5E1", "borderRadius": "8px",
                    "padding": "10px 16px", "cursor": "pointer",
                    "fontWeight": "600", "fontSize": "13px",
                },
            ),
        ], style={"display": "flex", "justifyContent": "flex-end",
                  "marginBottom": "18px"}),
        html.Div([
            scenario_number_input(
                "Scheduled staff", "sim-scheduled-staff", 12,
                "People originally scheduled for this period."),
            scenario_number_input(
                "Hours per scheduled staff member", "sim-shift-hours", 8,
                "Hours contributed per scheduled person.", max_value=24),
            scenario_number_input(
                "Assumed unavailable staff", "sim-unavailable", 2,
                "Hypothetical people unable to work this shift."),
            scenario_number_input(
                "Required coverage hours", "sim-required", 96,
                "Hours of staff coverage management needs."),
            scenario_number_input(
                "Backup staff", "sim-backup-staff", 1,
                "Additional backup people, separate from scheduled staff."),
            scenario_number_input(
                "Hours per backup staff member", "sim-backup-hours", 8,
                "Potential shift hours per backup person.", max_value=24),
        ], style={
            "display": "grid", "gridTemplateColumns":
            "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", "gap": "18px",
        }),
        html.Div([
            html.Label("Backup hours actually available (%)", style={
                "display": "block", "color": NAVY, "fontWeight": "600",
                "fontSize": "13px", "marginBottom": "12px",
            }),
            dcc.Slider(
                id="sim-backup-availability", min=0, max=100, step=5, value=100,
                marks={0: "0%", 50: "50%", 100: "100%"},
                tooltip={"placement": "bottom", "always_visible": False},
            ),
            html.P(
                "Example: 50% availability means only half of potential "
                "backup hours can be used during this period.",
                style={"color": GRAY, "fontSize": "12px"},
            ),
        ], style={"marginTop": "28px", "padding": "0 6px"}),
        html.P(id="sim-validation", style={
            "color": "#B45309", "fontSize": "13px", "marginTop": "10px",
        }),
    ], style={
        "backgroundColor": WHITE, "border": "1px solid #E2E8F0",
        "borderRadius": "12px", "padding": "24px", "marginTop": "16px",
    }),
    html.Div([
        html.Div(id="sim-gap-before", children="16.0 h"),
        html.Div(id="sim-gap-after", children="8.0 h"),
        html.Div(id="sim-gap-recovered", children="8.0 h"),
        html.Div(id="sim-coverage-pct", children="91.7%"),
    ], id="sim-metrics", style={
        "display": "flex", "gap": "16px", "flexWrap": "wrap",
        "marginTop": "20px", "marginBottom": "20px",
    }),
    chart_card(coverage_comparison_figure(compute_coverage_scenario(
        12, 8, 2, 96, 1, 8, 100
    )), "sim-coverage-chart"),
    html.Div([
        html.P(id="sim-interpretation", style={
            "color": NAVY, "lineHeight": "1.7", "marginBottom": "8px",
        }),
        html.P(
            "This is an illustrative capacity calculation, not an absence "
            "forecast, an absenteeism rate, a validated staffing recommendation, "
            "or a financial-loss estimate. All scenario inputs are hypothetical "
            "and independent of the historical dataset and exploration filters.",
            style={"color": GRAY, "fontSize": "13px", "lineHeight": "1.6"},
        ),
        html.Button(
            "Download Scenario (CSV)", id="sim-download-button", n_clicks=0,
            style={"backgroundColor": NAVY, "color": WHITE, "border": "none",
                   "padding": "12px 18px", "borderRadius": "8px",
                   "cursor": "pointer", "fontWeight": "600"},
        ),
        dcc.Download(id="sim-download"),
    ], style={
        "backgroundColor": WHITE, "border": "1px solid #E2E8F0",
        "borderRadius": "12px", "padding": "20px", "marginTop": "18px",
    }),

], style={"padding": "25px 0"})

# STEP 18: BUILD FIVE-TAB NAVIGATION
app.layout = html.Div([
    html.H1("Employee Absenteeism Analytics", style={
        "color": NAVY, "fontSize": "38px", "marginBottom": "10px"
    }),
    html.P("Understanding absence patterns and supporting "
           "evidence-based workforce planning.", style={
               "color": GRAY, "fontSize": "16px", "marginBottom": "30px"
           }),
    # STEP 19: GLOBAL EXPLORATION FILTERS
    # These affect the Overview and Patterns charts and KPIs.
    # Statistical model findings and business recommendations use
    # the unchanged full cohort, to avoid misleading comparisons.
    html.Div([
        html.Div([
            html.H3("Explore the Data", style={
                "color": NAVY, "margin": "0 0 6px"
            }),
            html.P("Choose any combination of filters. The Overview and "
                   "Absenteeism Patterns charts update together.",
                   style={"color": GRAY, "fontSize": "13px", "margin": "0"}),
        ], style={"marginBottom": "16px"}),
        html.Div([
            html.Div([
                html.Label("Month", style={"color": NAVY, "fontWeight": "600"}),
                dcc.Dropdown(
                    id="explore-months",
                    options=[{"label": calendar.month_name[i], "value": i}
                             for i in range(1, 13)],
                    value=[], multi=True, placeholder="All months",
                ),
            ]),
            html.Div([
                html.Label("Season", style={"color": NAVY, "fontWeight": "600"}),
                dcc.Dropdown(
                    id="explore-seasons",
                    options=[{"label": season_labels[i], "value": i}
                             for i in [2, 3, 1, 4]],
                    value=[], multi=True, placeholder="All seasons",
                ),
            ]),
            html.Div([
                html.Label("Absence reason", style={"color": NAVY, "fontWeight": "600"}),
                dcc.Dropdown(
                    id="explore-reasons",
                    options=[
                        {"label": reason_labels.get(int(code),
                          "No coded reason (0)" if int(code) == 0
                          else f"Reason code {int(code)}"),
                         "value": int(code)}
                        for code in sorted(df["Reason for absence"].unique())
                    ],
                    value=[], multi=True, placeholder="All reasons",
                ),
            ]),
            html.Div([
                html.Label("Duration", style={"color": NAVY, "fontWeight": "600"}),
                dcc.Dropdown(
                    id="explore-duration",
                    options=[
                        {"label": "All durations", "value": "all"},
                        {"label": "Zero hours", "value": "zero"},
                        {"label": "Shorter (1–15 hours)", "value": "short"},
                        {"label": "Extended (16+ hours)", "value": "extended"},
                    ], value="all", clearable=False,
                ),
            ]),
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "16px",
        }),
        html.Div([
            html.P(id="exploration-status", children="Showing all 740 records.",
                   style={"color": GRAY, "fontSize": "13px", "margin": "0"}),
            html.Button("Reset filters", id="explore-reset", n_clicks=0,
                        style={"backgroundColor": WHITE, "color": NAVY,
                               "border": "1px solid #CBD5E1", "borderRadius": "8px",
                               "padding": "9px 15px", "cursor": "pointer"}),
        ], style={"display": "flex", "alignItems": "center",
                  "justifyContent": "space-between", "flexWrap": "wrap",
                  "gap": "12px", "marginTop": "17px"}),
    ], style={"backgroundColor": WHITE, "padding": "20px",
              "border": "1px solid #E2E8F0", "borderRadius": "12px",
              "marginBottom": "24px"}),

    # Tabs serve as navigation only. Keep all page components mounted,
    # hiding inactive pages with CSS. This lets the filter callback safely
    # update charts even when their corresponding tab is not selected.
    dcc.Tabs(id="dashboard-tabs", value="overview", children=[
        dcc.Tab(label="Executive Overview", value="overview",
                style=tab_style, selected_style=selected_tab_style),
        dcc.Tab(label="Absenteeism Patterns", value="patterns",
                style=tab_style, selected_style=selected_tab_style),
        dcc.Tab(label="Factors & Evidence", value="factors",
                style=tab_style, selected_style=selected_tab_style),
        dcc.Tab(label="Workforce Simulator", value="simulator",
                style=tab_style, selected_style=selected_tab_style),
        dcc.Tab(label="Business Recommendations", value="recommendations",
                style=tab_style, selected_style=selected_tab_style),
    ]),
    html.Div(overview_content, id="overview-page", style={"display": "block"}),
    html.Div(patterns_content, id="patterns-page", style={"display": "none"}),
    html.Div(factors_content, id="factors-page", style={"display": "none"}),
    html.Div(simulator_content, id="simulator-page", style={"display": "none"}),
    html.Div(recommendations_content, id="recommendations-page",
             style={"display": "none"}),
], style={
    "fontFamily": "Arial, sans-serif", "backgroundColor": BACKGROUND,
    "padding": "40px 24px", "minHeight": "100vh",
    "maxWidth": "1400px", "margin": "0 auto", "boxSizing": "border-box",
})

# ============================================================
# STEP 19: LINKED FILTERING LOGIC AND CALLBACKS
# ============================================================

# STEP 19A: SHOW ONE NAVIGATION PAGE WITHOUT UNMOUNTING THE OTHERS.
# This design is important because a single filter callback updates
# figures on multiple tabs. CSS-hidden pages still exist in Dash's layout.
@app.callback(
    Output("overview-page", "style"),
    Output("patterns-page", "style"),
    Output("factors-page", "style"),
    Output("simulator-page", "style"),
    Output("recommendations-page", "style"),
    Input("dashboard-tabs", "value"),
)
def show_tab(selected_tab):
    return tuple(
        {"display": "block" if page == selected_tab else "none"}
        for page in ("overview", "patterns", "factors", "simulator", "recommendations")
    )




# User-selected filters are applied to *records*, not employees.
# The monthly and seasonal charts use only positive-hour rows
# when computing their distributions, as in the original notebook.
def apply_exploration_filters(months, seasons, reasons, duration):
    subset = df
    if months:
        subset = subset[subset["Month of absence"].isin(months)]
    if seasons:
        subset = subset[subset["Seasons"].isin(seasons)]
    if reasons:
        subset = subset[subset["Reason for absence"].isin(reasons)]
    hours = subset["Absenteeism time in hours"]
    if duration == "zero":
        subset = subset[hours == 0]
    elif duration == "short":
        subset = subset[(hours > 0) & (hours < 16)]
    elif duration == "extended":
        subset = subset[hours >= 16]
    return subset.copy()


def blank_figure(title, height=440):
    """Return a readable chart when the selection contains no data."""
    figure = go.Figure()
    figure.update_layout(
        title=title, height=height, paper_bgcolor=WHITE,
        plot_bgcolor=WHITE, font=dict(color=NAVY),
        annotations=[dict(text="No records match the current filters",
                          x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=15, color=GRAY))],
    )
    return figure


def filtered_reasons_figure(positive):
    if positive.empty:
        return blank_figure("Top 10 Reasons by Recorded Absence Hours", 550)
    summary = positive.groupby("Reason for absence").agg(
        records=("Absenteeism time in hours", "size"),
        total_hours=("Absenteeism time in hours", "sum"),
    ).reset_index().nlargest(10, "total_hours")
    summary["Reason"] = summary["Reason for absence"].map(
        lambda code: reason_labels.get(int(code), f"Reason code {int(code)}")
    )
    summary = summary.sort_values("total_hours")
    fig = px.bar(
        summary, x="total_hours", y="Reason", orientation="h",
        title="Top 10 Reasons by Recorded Absence Hours",
        text="total_hours", custom_data=["records"],
        color_discrete_sequence=[TEAL],
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Total Hours: %{x}"
                      "<br>Absence Records: %{customdata[0]}<extra></extra>",
    )
    fig.update_layout(
        height=550, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(color=NAVY), xaxis_title="Total Recorded Absence Hours",
        yaxis_title="", margin=dict(l=20, r=35, t=70, b=50),
    )
    return fig


def filtered_month_figures(positive):
    valid = positive[positive["Month of absence"].between(1, 12)]
    monthly = valid.groupby("Month of absence").agg(
        records=("Absenteeism time in hours", "size"),
        total_hours=("Absenteeism time in hours", "sum"),
    ).reindex(range(1, 13), fill_value=0).reset_index()
    monthly["Month"] = monthly["Month of absence"].map(
        lambda m: calendar.month_name[m]
    )
    h = px.bar(
        monthly, x="Month", y="total_hours", text="total_hours",
        title="Recorded Absence Hours by Month",
        category_orders={"Month": month_order}, color_discrete_sequence=[TEAL],
    )
    h.update_traces(textposition="outside")
    h.update_layout(height=440, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                    font=dict(color=NAVY), xaxis_title="Month",
                    yaxis_title="Recorded Absence Hours",
                    margin=dict(l=30, r=20, t=75, b=90))
    h.update_xaxes(tickangle=-45)
    r = px.line(
        monthly, x="Month", y="records", markers=True,
        title="Number of Absence Records by Month",
        category_orders={"Month": month_order}, color_discrete_sequence=[PURPLE],
    )
    r.update_traces(line=dict(width=3), marker=dict(size=9))
    r.update_layout(height=440, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                    font=dict(color=NAVY), xaxis_title="Month",
                    yaxis_title="Number of Absence Records",
                    margin=dict(l=30, r=20, t=75, b=90))
    r.update_xaxes(tickangle=-45)
    return h, r


def filtered_duration_figure(subset, duration):
    if subset.empty:
        return blank_figure("Distribution of Recorded Absence Durations", 470)
    fig = px.histogram(
        subset, x="Absenteeism time in hours",
        title="Distribution of Recorded Absence Durations",
        color_discrete_sequence=[PURPLE],
    )
    fig.update_traces(xbins=dict(start=-0.5, end=120.5, size=1))
    # Widen the view for extended-only selections so that hours
    # beyond 16 are not hidden by the original zoom setting.
    if duration == "extended":
        fig.update_xaxes(range=[15.5, max(24.5, float(subset[
            "Absenteeism time in hours"].max()) + 0.5)])
    else:
        fig.update_xaxes(range=[-0.5, 16.5], dtick=2)
    fig.update_layout(height=470, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                      font=dict(color=NAVY), xaxis_title="Absence Duration (Hours)",
                      yaxis_title="Number of Records",
                      margin=dict(l=35, r=20, t=75, b=60))
    return fig


def filtered_weekday_figure(positive):
    valid = positive[positive["Day of the week"].isin(weekday_labels)].copy()
    valid["extended_hours"] = valid["Absenteeism time in hours"].where(
        valid["Absenteeism time in hours"] >= 16, 0
    )
    weekday = valid.groupby("Day of the week").agg(
        total_hours=("Absenteeism time in hours", "sum"),
        extended_hours=("extended_hours", "sum"),
    ).reindex([2, 3, 4, 5, 6], fill_value=0).reset_index()
    weekday["Weekday"] = weekday["Day of the week"].map(weekday_labels)
    weekday["shorter_hours"] = weekday["total_hours"] - weekday["extended_hours"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=weekday["Weekday"], y=weekday["shorter_hours"],
        name="Shorter (<16 Hours)", marker_color=PURPLE,
        text=weekday["shorter_hours"], textposition="inside",
    ))
    fig.add_trace(go.Bar(
        x=weekday["Weekday"], y=weekday["extended_hours"],
        name="Extended (16+ Hours)", marker_color=TEAL,
        text=weekday["extended_hours"], textposition="inside",
    ))
    fig.update_layout(
        title="Shorter vs. Extended Absences by Weekday", barmode="stack",
        height=470, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
        font=dict(color=NAVY), xaxis_title="Weekday",
        yaxis_title="Recorded Absence Hours", legend_title_text="Absence Duration",
        legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(l=40, r=20, t=75, b=100),
    )
    return fig


def filtered_seasons_figure(positive):
    if positive.empty:
        return blank_figure("Seasonal Absence Frequency vs. Total Hours", 460)
    valid = positive[positive["Seasons"].isin(season_labels)]
    summary = valid.groupby("Seasons").agg(
        records=("Absenteeism time in hours", "size"),
        total_hours=("Absenteeism time in hours", "sum"),
    ).reindex([2, 3, 1, 4], fill_value=0).reset_index()
    summary["Season"] = summary["Seasons"].map(season_labels)
    count_denominator = summary["records"].sum()
    hour_denominator = summary["total_hours"].sum()
    summary["Record Share (%)"] = (
        summary["records"] / count_denominator * 100
        if count_denominator else 0.0
    )
    summary["Hour Share (%)"] = (
        summary["total_hours"] / hour_denominator * 100
        if hour_denominator else 0.0
    )
    long = summary.melt(id_vars="Season",
                        value_vars=["Record Share (%)", "Hour Share (%)"],
                        var_name="Metric", value_name="Percentage")
    fig = px.bar(
        long, x="Season", y="Percentage", color="Metric", barmode="group",
        title="Seasonal Absence Frequency vs. Total Hours",
        category_orders={"Season": ["Winter", "Spring", "Summer", "Fall"]},
        color_discrete_map={"Record Share (%)": PURPLE, "Hour Share (%)": TEAL},
        text="Percentage",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    max_share = max(float(long["Percentage"].max()), 0)
    y_max = max(35, min(110, max_share * 1.2))
    fig.update_layout(height=460, paper_bgcolor=WHITE, plot_bgcolor=WHITE,
                      font=dict(color=NAVY), xaxis_title="Season",
                      yaxis_title="Share Within Selected Records (%)",
                      yaxis_range=[0, y_max], legend_title_text="Measurement",
                      margin=dict(l=50, r=30, t=80, b=55))
    return fig


def fraction_of_hours(numerator, denominator):
    """Return a formatted share, or a dash when there are no hours."""
    return f"{numerator / denominator * 100:.1f}%" if denominator else "—"


@app.callback(
    Output("overview-filtered-kpis", "children"),
    Output("overview-reasons-chart", "figure"),
    Output("overview-monthly-hours-chart", "figure"),
    Output("overview-monthly-records-chart", "figure"),
    Output("patterns-filtered-kpis", "children"),
    Output("patterns-duration-chart", "figure"),
    Output("patterns-weekday-chart", "figure"),
    Output("patterns-seasons-chart", "figure"),
    Output("exploration-status", "children"),
    Input("explore-months", "value"),
    Input("explore-seasons", "value"),
    Input("explore-reasons", "value"),
    Input("explore-duration", "value"),
)
def update_exploration(months, seasons, reasons, duration):
    # Dash passes current selections; use one filtered DataFrame
    # across all linked KPI cards and charts for consistency.
    selected = apply_exploration_filters(months, seasons, reasons, duration)
    positive = selected[selected["Absenteeism time in hours"] > 0]
    n = len(selected)
    employee_count = selected["ID"].nunique()
    hours_total = int(selected["Absenteeism time in hours"].sum())
    positive_count = len(positive)

    overview_kpis = [
        kpi_card("Total Records", f"{n:,}", "Records in current selection"),
        kpi_card("Unique Employees", f"{employee_count:,}",
                 "Distinct IDs in current selection"),
        kpi_card("Recorded Absence Hours", f"{hours_total:,}",
                 "Total hours in current selection"),
        kpi_card("Positive-Hour Records", f"{positive_count:,}",
                 "Positive-hour records in current selection"),
    ]

    extended = positive.loc[
        positive["Absenteeism time in hours"] >= 16,
        "Absenteeism time in hours"
    ].sum()
    early = positive.loc[
        positive["Day of the week"].isin([2, 3, 4]),
        "Absenteeism time in hours"
    ].sum()
    spring = positive.loc[
        positive["Seasons"] == 3,
        "Absenteeism time in hours"
    ].sum()
    pattern_kpis = [
        kpi_card("Extended Absence Hours", fraction_of_hours(extended, hours_total),
                 "Share of selected hours from 16+ hour records"),
        kpi_card("Monday–Wednesday Hours", fraction_of_hours(early, hours_total),
                 "Share of selected hours with early-week codes"),
        kpi_card("Spring Absence Hours", fraction_of_hours(spring, hours_total),
                 "Share of selected recorded hours"),
    ]

    monthly_hours, monthly_records = filtered_month_figures(positive)
    status = (
        f"Showing {n:,} of {total_records:,} records, "
        f"{employee_count} distinct employees and {hours_total:,} recorded hours. "
        "The Factors & Evidence and Business Recommendations tabs "
        "continue to show the full historical cohort."
    )
    return (
        overview_kpis,
        filtered_reasons_figure(positive),
        monthly_hours, monthly_records,
        pattern_kpis,
        filtered_duration_figure(selected, duration),
        filtered_weekday_figure(positive),
        filtered_seasons_figure(positive),
        status,
    )


@app.callback(
    Output("explore-months", "value"),
    Output("explore-seasons", "value"),
    Output("explore-reasons", "value"),
    Output("explore-duration", "value"),
    Input("explore-reset", "n_clicks"),
    prevent_initial_call=True,
)
def reset_exploration(n_clicks):
    # Return each input to the no-filter state.
    return [], [], [], "all"


# STEP 18A: INTERACTIVE RECOMMENDATION FILTER
# A Dash callback updates the displayed cards whenever the
# user chooses a different proposed priority in the dropdown.
@app.callback(
    Output("recommendation-card-grid", "children"),
    Input("recommendation-priority-filter", "value"),
)
def filter_recommendations(selected_priority):
    if selected_priority == "All":
        selected_rows = recommendation_rows
    else:
        selected_rows = [
            item for item in recommendation_rows
            if item["Priority"] == selected_priority
        ]
    return [recommendation_card(item) for item in selected_rows]


# STEP 18B: AGGREGATE CSV DOWNLOAD
# Download contains only summary evidence and recommended
# actions; it never exports employee-level health data.
@app.callback(
    Output("recommendations-download", "data"),
    Input("download-recommendations-button", "n_clicks"),
    prevent_initial_call=True,
)
def download_recommendations(n_clicks):
    return dcc.send_data_frame(
        recommendations_df.to_csv,
        "absenteeism_business_recommendations.csv",
        index=False,
    )


# ============================================================
# STEP 21B: FILTERED EXPORTS AND AUDITABLE ANALYSIS PACK
# ============================================================
# Exports use the same apply_exploration_filters function as the chart
# callbacks. Summaries include all available categories, not just top 10.
# Positive-hour records determine month, reason, weekday and season
# totals. Zero-hour records remain visible in selection and detailed CSV.

def filtered_detail_export(selected):
    """Record-level export without IDs or personal profile columns.

    A record-level export is not guaranteed to be anonymous, especially
    in a small dataset; use only within appropriate data-sharing rules.
    """
    columns = [
        "Month of absence", "Day of the week", "Seasons",
        "Reason for absence", "Absenteeism time in hours",
    ]
    detail = selected.loc[:, columns].copy().reset_index(drop=True)
    detail.insert(0, "Export row", range(1, len(detail) + 1))
    detail["Month name"] = detail["Month of absence"].map(
        lambda month: calendar.month_name[int(month)] if 1 <= int(month) <= 12
        else "Unknown"
    )
    detail["Weekday name"] = detail["Day of the week"].map(weekday_labels)
    detail["Season name"] = detail["Seasons"].map(season_labels)
    detail["Reason label"] = detail["Reason for absence"].map(
        lambda code: reason_labels.get(
            int(code), "No coded reason (0)" if int(code) == 0
            else f"Reason code {int(code)}"
        )
    )
    detail["Duration category"] = detail["Absenteeism time in hours"].map(
        lambda hours: "Zero hours" if hours == 0
        else "Extended (16+ hours)" if hours >= 16
        else "Shorter (1-15 hours)"
    )
    return detail


def analysis_export_frames(selected, months, seasons, reasons, duration):
    """Generate internally consistent reports for one filter selection."""
    positive = selected[selected["Absenteeism time in hours"] > 0].copy()
    hours_column = "Absenteeism time in hours"
    selected_hours = int(selected[hours_column].sum())

    # Selection report includes both all records and positive-hour records.
    summary = pd.DataFrame([
        ("Selected months", ", ".join(str(m) for m in (months or [])) or "All"),
        ("Selected seasons", ", ".join(season_labels.get(int(s), str(s))
                                      for s in (seasons or [])) or "All"),
        ("Selected reasons", ", ".join(str(r) for r in (reasons or [])) or "All"),
        ("Selected duration", duration or "all"),
        ("Total selected records (including zero-hour)", len(selected)),
        ("Selected positive-hour records", len(positive)),
        ("Distinct employee IDs in selected records", selected["ID"].nunique()),
        ("Recorded hours in selection", selected_hours),
        ("Extended-hour records in selection", int((positive[hours_column] >= 16).sum())),
        ("Recorded hours from extended-hour records", int(
            positive.loc[positive[hours_column] >= 16, hours_column].sum()
        )),
    ], columns=["Metric", "Value"])

    # A full calendar summary remains sortable even for empty selections.
    monthly = (
        positive[positive["Month of absence"].between(1, 12)]
        .groupby("Month of absence")
        .agg(records=(hours_column, "size"), total_hours=(hours_column, "sum"))
        .reindex(range(1, 13), fill_value=0).reset_index()
    )
    monthly.insert(1, "Month", monthly["Month of absence"].map(
        lambda m: calendar.month_name[int(m)]
    ))

    reason = (positive.groupby("Reason for absence")
              .agg(records=(hours_column, "size"),
                   total_hours=(hours_column, "sum"),
                   avg_hours=(hours_column, "mean"))
              .reset_index().sort_values("total_hours", ascending=False))
    reason.insert(1, "Reason", reason["Reason for absence"].map(
        lambda code: reason_labels.get(
            int(code), f"Reason code {int(code)}"
        )
    ))
    reason["avg_hours"] = reason["avg_hours"].round(2)

    weekday = (
        positive[positive["Day of the week"].isin(weekday_labels)]
        .assign(extended_hours=lambda frame: frame[hours_column].where(
            frame[hours_column] >= 16, 0
        ))
        .groupby("Day of the week")
        .agg(records=(hours_column, "size"),
             total_hours=(hours_column, "sum"),
             extended_hours=("extended_hours", "sum"))
        .reindex([2, 3, 4, 5, 6], fill_value=0).reset_index()
    )
    weekday.insert(1, "Weekday", weekday["Day of the week"].map(weekday_labels))
    weekday["shorter_hours"] = weekday["total_hours"] - weekday["extended_hours"]

    seasonal = (positive[positive["Seasons"].isin(season_labels)]
                .groupby("Seasons")
                .agg(records=(hours_column, "size"), total_hours=(hours_column, "sum"))
                .reindex([2, 3, 1, 4], fill_value=0).reset_index())
    seasonal.insert(1, "Season", seasonal["Seasons"].map(season_labels))
    total_positive = len(positive)
    seasonal["record_share_pct"] = (
        (seasonal["records"] / total_positive * 100).round(2)
        if total_positive else 0.0
    )
    seasonal["hour_share_pct"] = (
        (seasonal["total_hours"] / selected_hours * 100).round(2)
        if selected_hours else 0.0
    )

    durations = (selected.groupby(hours_column)
                 .size().reset_index(name="records")
                 .sort_values(hours_column))
    durations["total_hours"] = durations[hours_column] * durations["records"]

    limitations = pd.DataFrame({"Notes": [
        "All tables except record-level detail and selection summary describe positive-hour records.",
        "Report totals follow the selected filters and do not represent absenteeism rates.",
        "The dataset lacks exact absence dates, year identifiers and scheduled hours.",
        "One recorded absence can span multiple days but is associated with one weekday code.",
        "Season labels are inferred from the month-to-season data cross-tab.",
        "Extended means at least 16 recorded hours; it is an analytical threshold.",
        "No employee ID or personal profile columns are included in the detail export.",
        "Record-level export is not guaranteed anonymous; review privacy rules before distributing.",
        "Management recommendation CSV concerns the full historical cohort, not current filters.",
        "The regression did not outperform the training-mean baseline in employee-level LOOCV.",
        "The staffing simulator uses hypothetical inputs, not observed workforce schedules.",
    ]})

    return {
        "00_selected_kpis.csv": summary,
        "01_filtered_records_no_ids.csv": filtered_detail_export(selected),
        "02_monthly_summary.csv": monthly,
        "03_reason_summary.csv": reason,
        "04_weekday_summary.csv": weekday,
        "05_seasonal_summary.csv": seasonal,
        "06_duration_summary.csv": durations,
        "07_recommendations_FULL_COHORT.csv": recommendations_df.copy(),
        "08_methods_and_limitations.csv": limitations,
    }


@app.callback(
    Output("filtered-records-download", "data"),
    Input("download-filtered-records-button", "n_clicks"),
    State("explore-months", "value"),
    State("explore-seasons", "value"),
    State("explore-reasons", "value"),
    State("explore-duration", "value"),
    prevent_initial_call=True,
)
def download_filtered_records(n_clicks, months, seasons, reasons, duration):
    selected = apply_exploration_filters(months, seasons, reasons, duration)
    return dcc.send_data_frame(
        filtered_detail_export(selected).to_csv,
        "absenteeism_filtered_records_no_ids.csv", index=False,
    )


@app.callback(
    Output("analysis-pack-download", "data"),
    Input("download-analysis-pack-button", "n_clicks"),
    State("explore-months", "value"),
    State("explore-seasons", "value"),
    State("explore-reasons", "value"),
    State("explore-duration", "value"),
    prevent_initial_call=True,
)
def download_analysis_pack(n_clicks, months, seasons, reasons, duration):
    selected = apply_exploration_filters(months, seasons, reasons, duration)
    frames = analysis_export_frames(selected, months, seasons, reasons, duration)

    # dcc.send_bytes invokes this writer with an in-memory output stream.
    # All included CSVs carry a UTF-8 BOM for Excel compatibility.
    def write_zip(stream):
        with ZipFile(stream, mode="w", compression=ZIP_DEFLATED) as archive:
            for filename, table in frames.items():
                archive.writestr(filename, "\ufeff" + table.to_csv(index=False))
    return dcc.send_bytes(write_zip, "absenteeism_analysis_pack.zip")


# STEP 22C: ONE-CLICK SIMULATOR RESET
# Restoring the seven input values automatically triggers the existing
# live-simulation callback. Historical exploration filters are unchanged.
@app.callback(
    Output("sim-scheduled-staff", "value"),
    Output("sim-shift-hours", "value"),
    Output("sim-unavailable", "value"),
    Output("sim-required", "value"),
    Output("sim-backup-staff", "value"),
    Output("sim-backup-hours", "value"),
    Output("sim-backup-availability", "value"),
    Input("sim-reset-button", "n_clicks"),
    prevent_initial_call=True,
)
def reset_coverage_scenario(n_clicks):
    """Reset assumptions to defaults without changing dashboard filters."""
    if not n_clicks:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    return (12, 8, 2, 96, 1, 8, 100)


# ============================================================
# STEP 20C: LIVE SCENARIO CALCULATION AND EXPORT
# ============================================================
# Each change to a scenario control recalculates staffing
# capacity for that scenario; the historical dataset is
# NEVER used to invent scheduled shifts or absence likelihood.

@app.callback(
    Output("sim-gap-before", "children"),
    Output("sim-gap-after", "children"),
    Output("sim-gap-recovered", "children"),
    Output("sim-coverage-pct", "children"),
    Output("sim-coverage-chart", "figure"),
    Output("sim-interpretation", "children"),
    Output("sim-validation", "children"),
    Input("sim-scheduled-staff", "value"),
    Input("sim-shift-hours", "value"),
    Input("sim-unavailable", "value"),
    Input("sim-required", "value"),
    Input("sim-backup-staff", "value"),
    Input("sim-backup-hours", "value"),
    Input("sim-backup-availability", "value"),
)
def update_coverage_simulator(staff, shift_hours, unavailable, needed_hours,
                              backup_staff, backup_hours, availability_pct):
    try:
        r = compute_coverage_scenario(
            staff, shift_hours, unavailable, needed_hours,
            backup_staff, backup_hours, availability_pct
        )
    except ValueError as exc:
        return ("—", "—", "—", "—",
                blank_figure("Hypothetical Usable Coverage Hours"),
                "Enter valid assumptions to view this scenario.", str(exc))

    before = kpi_card("Gap without backup",
                      f"{r['Coverage gap without backup']:,.1f} h",
                      "Required hours not covered by remaining staff")
    after = kpi_card("Gap with backup",
                     f"{r['Coverage gap with backup']:,.1f} h",
                     "Unfilled required hours after backup availability")
    recovered = kpi_card("Gap reduced by backup",
                         f"{r['Coverage gap reduced by']:,.1f} h",
                         "Coverage gap avoided, within required hours")
    share = kpi_card("Coverage of required hours",
                     f"{r['Covered share of required hours (%)']:.1f}%",
                     "Capped at 100%; not an absenteeism rate")
    explanation = (
        f"The scenario starts with {r['Originally scheduled staff hours']:,.1f} "
        f"scheduled staff-hours. With {r['Unavailable scheduled staff']} "
        f"assumed unavailable staff, {r['Available hours without backup']:,.1f} "
        f"regular hours remain. Backup adds {r['Effective backup hours']:,.1f} "
        f"usable hours toward {r['Required coverage hours']:,.1f} required "
        f"hours. The hypothetical unfilled gap changes from "
        f"{r['Coverage gap without backup']:,.1f} to "
        f"{r['Coverage gap with backup']:,.1f} hours."
    )
    return (before, after, recovered, share,
            coverage_comparison_figure(r), explanation, "")


@app.callback(
    Output("sim-download", "data"),
    Input("sim-download-button", "n_clicks"),
    State("sim-scheduled-staff", "value"),
    State("sim-shift-hours", "value"),
    State("sim-unavailable", "value"),
    State("sim-required", "value"),
    State("sim-backup-staff", "value"),
    State("sim-backup-hours", "value"),
    State("sim-backup-availability", "value"),
    prevent_initial_call=True,
)
def download_coverage_scenario(n_clicks, staff, shift_hours, unavailable,
                               needed_hours, backup_staff, backup_hours,
                               availability_pct):
    # Don't export invalid assumptions. The UI explains how
    # to correct them using the live validation message.
    from dash.exceptions import PreventUpdate
    try:
        results = compute_coverage_scenario(
            staff, shift_hours, unavailable, needed_hours,
            backup_staff, backup_hours, availability_pct
        )
    except ValueError:
        raise PreventUpdate
    results["Scenario type"] = "Illustrative inputs; not historical staffing data"
    return dcc.send_data_frame(
        pd.DataFrame([results]).to_csv,
        "workforce_coverage_scenario.csv", index=False
    )


# STEP 21C: RUN LOCALLY OR EXPOSE A WSGI APP FOR DEPLOYMENT
# For local development: python app.py  (auto reload enabled).
# For a Linux deployment: gunicorn app:server --bind 0.0.0.0:$PORT
# Do not use the local debug server in production.
if __name__ == "__main__":
    app.run(debug=True)
