# Publish the project to GitHub

The repository name suggestion is **`employee-absenteeism-analytics`**.

Suggested GitHub description:

> Interactive Python Dash analytics application with absenteeism trends, statistical validation, workforce coverage scenarios, and downloadable reports.

Suggested topics: `python`, `dash`, `plotly`, `data-analytics`, `workforce-analytics`, `data-visualization`, `scikit-learn`.

## Before publishing

1. Run the application and check all five tabs, both reset controls, and downloads.
2. Run `python -m unittest discover -s tests -v` and address any failures.
3. Confirm the repo contains only this **public UCI sample dataset**, not any real employee data, passwords, tokens, patient records, or local `.venv` files.
4. Add a real screenshot later; don't use an unverified mockup as a dashboard screenshot.

## Option A — GitHub website (no Git commands required)

1. Sign in to https://github.com/ and choose **New repository**.
2. Name it `employee-absenteeism-analytics`, choose **Public** when ready, and create it **without** automatically adding a README or `.gitignore` (both already exist in the project).
3. Open the new repository and choose **Add file → Upload files**.
4. **Extract the ZIP first** and upload the project files and folders, not the ZIP as one attachment. Preserve the `Data` and `tests` folders and the `.github/workflows` hierarchy.
5. Commit the upload. Open the README on GitHub and make sure the formatting renders as expected.

## Option B — VS Code terminal (Git installed)

From the extracted project root, use:

```powershell
git init
git branch -M main
git add .
git commit -m "Add employee absenteeism analytics dashboard"
git remote add origin https://github.com/YOUR_USERNAME/employee-absenteeism-analytics.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If GitHub asks you to authenticate, follow GitHub's sign-in or credential flow; do not paste tokens into repository files.

## After uploading

The **Actions** tab should run `Dashboard tests` automatically on the `main` branch. A passing workflow is helpful, but it does not replace a real browser deployment test. Follow [DEPLOYMENT.md](DEPLOYMENT.md) for the online version.
