# Employee Absenteeism Analytics Dashboard

An interactive Python **Dash** application that analyzes historical workplace absence records and supports evidence-based workforce planning. It combines exploratory analytics, model validation, practical recommendations, downloadable reports, and a **hypothetical workforce coverage simulator**.

> **Project status:** Working locally; prepared for GitHub and cloud deployment. A public deployment URL has not yet been established.

## Highlights

- **Five dashboard tabs:** Executive Overview; Absenteeism Patterns; Factors & Evidence; Workforce Simulator; Business Recommendations.
- **Linked data filters:** month, season, absence reason, and duration, with a reset option. The filters drive the overview and patterns charts.
- **Statistical validation:** exploratory associations, regression diagnostics, and leave-one-employee-out evaluation against a baseline.
- **Scenario simulator:** change staff counts, shift length, assumed unavailability, coverage requirements, backup capacity, and availability. Review the calculated coverage gap, visualize results, export a scenario, or reset to defaults.
- **Downloadable reports:** filtered observation CSV (without employee ID and personal-profile fields), nine-report analysis ZIP, management recommendations, and hypothetical scenario CSV.
- **Automated tests:** dataset totals, filtering edge cases, export consistency, five-tab navigation, simulator calculation and reset.

## Preview

Screenshots will be added after the final browser review. The application currently runs locally in a Dash browser session.

## Data and analytical findings

The project uses the [UCI *Absenteeism at work* dataset](https://archive.ics.uci.edu/dataset/445/absenteeism), a publicly available research dataset. The source file contains **740 observations** and **21 columns**, representing **36 coded employees** and **5,124 recorded absence hours**; **696 records** have positive absence hours. The CSV is semicolon-delimited.

Selected descriptive findings:

| Measure | Finding |
| --- | ---: |
| Total recorded absence hours | 5,124 |
| Positive-hour records | 696 |
| Extended records (16+ hours) | 63 |
| Hours in extended records | 2,440 (47.6% of hours) |
| Recorded hours in March | 765 |

**Interpretation matters:** observed absence hours are not an *absenteeism rate* because the data does not provide scheduled working hours. The exploratory employee-level regression does **not** outperform the baseline in held-out validation, so the application does not claim reliable individual absence prediction. Associations are not causal evidence, and the simulator does not learn staffing forecasts from the dataset.

## Tech stack

Python · pandas · NumPy · SciPy · scikit-learn · statsmodels · Plotly · Dash · Gunicorn (Linux deployment)

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

If you already have a working virtual environment, activate it and skip the `venv` creation step.

## Run automated tests

```powershell
python -m unittest discover -s tests -v
```

The test suite uses the real source CSV and imports the Dash app, so first install the dependencies.

## Project structure

```text
employee-absenteeism-analytics/
├── app.py                       # Dash application; five tabs and callbacks
├── Data/
│   └── Absenteeism_at_work.csv  # UCI sample dataset
├── tests/
│   └── test_dashboard.py        # Calculation, filter, export, and reset checks
├── .github/workflows/tests.yml  # Automated tests on GitHub
├── requirements.txt             # Python dependencies
├── Procfile                     # Optional host startup command
├── .gitignore                   # Ignore virtual envs, caches, and secrets
├── DATA_ATTRIBUTION.md          # Data attribution and license
├── GITHUB_SETUP.md              # Repository publication steps
├── DEPLOYMENT.md                # Render web service setup
└── README.md
```

## Privacy, scope and limitations

- The source is a *public research dataset*, not operational data from the user's workplace. Do not substitute identifiable employee medical records without privacy review and appropriate authorization.
- Removing IDs from a row-level export **does not guarantee anonymity**; source characteristics and combinations may remain sensitive. Check data sharing obligations before publishing or reusing exports.
- The underlying records do not support a true absenteeism rate, causal claims, or validated employee-level predictions.
- The **Factors & Evidence** and **Business Recommendations** pages describe the full historical cohort, even when global exploration filters are active.
- The simulator reflects **user-entered hypothetical coverage assumptions**, not observed schedules, costs, or forecasted absence probabilities.
- The 16-hour extended-duration threshold is an analytical convention, not a universal business standard.

## Dataset attribution

> Martiniano, A. & Ferreira, R. (2012). *Absenteeism at work* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5X882

The UCI dataset is shared under **Creative Commons Attribution 4.0 (CC BY 4.0)**. See [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md) for the citation and license link. The dataset license applies to the data, not automatically to this dashboard's original source code.

## Publishing and deployment

Follow [GITHUB_SETUP.md](GITHUB_SETUP.md) to upload this repository and [DEPLOYMENT.md](DEPLOYMENT.md) to host it. **GitHub hosting alone will not run this Python Dash app**; a separate Python web service is needed for an interactive public URL.
