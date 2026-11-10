# Employee Absenteeism Analytics Dashboard

**[Launch Live Dashboard](https://employee-absenteeism-analytics.onrender.com/)** · **[View Source on GitHub](https://github.com/Selena1224/employee_absenteeism_analytics)**

> **Loading note:** The dashboard may take about a minute to load after a period of inactivity while the server starts. Please keep the page open.

An interactive Python **Dash** application that analyzes historical workplace absence records and supports evidence-informed workforce planning. It combines exploratory analytics, model validation, practical recommendations, downloadable reports, and a **hypothetical workforce coverage simulator**.

**Project status:** Deployed publicly on Render. The dashboard's live interactions have been tested by its creator.

## Highlights

- **Five dashboard tabs:** Executive Overview; Absenteeism Patterns; Factors & Evidence; Workforce Simulator; Business Recommendations.
- **Linked data filters:** month, season, absence reason, and duration, with a reset option. The filters drive the overview and patterns charts.
- **Statistical validation:** exploratory associations, regression diagnostics, and leave-one-employee-out evaluation against a baseline.
- **Workforce coverage simulator:** change staff counts, hours per person, assumed unavailability, coverage requirements, backup capacity, and availability. See the resulting coverage gap, compare scenarios visually, export results, and reset to defaults.
- **Downloadable reports:** filtered observation CSV (without employee ID and personal-profile fields), nine-report analysis ZIP, management recommendations, and hypothetical scenario CSV.
- **Automated tests:** dataset totals, filtering edge cases, export consistency, five-tab navigation, simulator calculations, and reset behavior.

## Live demo

**[Open the Interactive Dashboard](https://employee-absenteeism-analytics.onrender.com/)**

The project includes interactive filtering, visualizations, and a hypothetical staffing scenario tool. Screenshots can be added here for a quick preview without waiting for the live application to start.

## Data and analytical findings

The project uses the [UCI *Absenteeism at work* dataset](https://archive.ics.uci.edu/dataset/445/absenteeism), a public research dataset. The source file contains **740 observations** and **21 columns**, representing **36 coded employees** and **5,124 recorded absence hours**; **696 records** have positive absence hours. The CSV is semicolon-delimited.

Selected descriptive findings:

| Measure | Finding |
| --- | ---: |
| Total recorded absence hours | 5,124 |
| Positive-hour records | 696 |
| Extended records (16+ hours) | 63 |
| Hours in extended records | 2,440 (47.6% of hours) |
| Recorded hours in March | 765 |

**Interpretation matters:** recorded absence hours are not an *absenteeism rate* because the dataset does not provide scheduled working hours. The exploratory employee-level regression did **not** outperform the baseline in held-out validation, so the application does not claim reliable individual absence prediction. Associations are not causal evidence, and the simulator does not forecast absences from the dataset.

## Tech stack

Python · pandas · NumPy · SciPy · scikit-learn · statsmodels · Plotly · Dash · Gunicorn · Render

## Run locally

Use Python 3.11 or another version compatible with `requirements.txt`.

```powershell
# Windows PowerShell — from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:8050/**.

If you already have a working virtual environment, activate it and skip virtual-environment creation.

## Run automated tests

```powershell
python -m unittest discover -s tests -v
```

The test suite uses the source CSV and imports the Dash app, so install dependencies first.

## Repository structure

```text
employee_absenteeism_analytics/
├── app.py                       # Dash application, five tabs, and callbacks
├── Data/
│   └── Absenteeism_at_work.csv  # UCI research dataset
├── tests/
│   └── test_dashboard.py        # Calculations, filters, exports, and reset
├── requirements.txt             # Python dependencies
├── Procfile                     # Web service startup command
├── DATA_ATTRIBUTION.md          # Dataset attribution and license
├── GITHUB_SETUP.md              # GitHub setup guidance
├── DEPLOYMENT.md                # Render deployment guidance
└── README.md
```

## Privacy, scope, and limitations

- The source is a *public research dataset*, not operational data from an employer. Identifiable employee medical records should not be substituted without appropriate authorization and privacy review.
- Removing IDs from a row-level export **does not guarantee anonymity**; data combinations may still be identifiable. Review data-sharing obligations before reusing exports with other data.
- The records do not support a true absenteeism rate, causal claims, or validated employee-level predictions.
- The **Factors & Evidence** and **Business Recommendations** pages describe the full historical cohort, even when exploration filters are active.
- The workforce simulator uses **user-entered hypothetical staffing assumptions**, not observed schedules, costs, or forecasted absence probabilities.
- The 16-hour extended-duration threshold is an analytical convention, not a universal business standard.

## Dataset attribution

> Martiniano, A. & Ferreira, R. (2012). *Absenteeism at work* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X882

The dataset is distributed under **Creative Commons Attribution 4.0 (CC BY 4.0)**. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the citation and license link. The dataset license applies to the data, not automatically to this dashboard's original source code.

## Deployment

The application is publicly hosted on **Render**:

**https://employee-absenteeism-analytics.onrender.com/**

For instructions to redeploy or run a similar Python web service, see [DEPLOYMENT.md](DEPLOYMENT.md). A GitHub repository hosts the source code; Render runs the interactive Dash server.
