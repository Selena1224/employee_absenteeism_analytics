# Deploy the Dash app as a live website

GitHub displays and versions the code; it does **not** execute an interactive Dash server. This project can be deployed to a Python web host. The steps below use **Render** as one option.

## Prerequisites

- A working GitHub repository containing all of this project's files.
- Successful local `python app.py` test.
- A compatible Python runtime on the host and dependencies from `requirements.txt`.

## Render web service

1. Visit https://dashboard.render.com/ and create **New → Web Service**.
2. Connect your GitHub repository `employee-absenteeism-analytics`.
3. Choose **Python 3** runtime.
4. **Build command:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Start command:**

   ```bash
   gunicorn app:server --bind 0.0.0.0:$PORT
   ```

   `server = app.server` is already defined in `app.py`, enabling Gunicorn to serve Dash's underlying Flask server.

6. Select the region and service plan appropriate for the project. Review the current Render pricing/limits before choosing a plan.
7. Deploy and wait for the service to build. Render provides a public `onrender.com` URL after a successful deployment.

## Browser smoke test after deployment

- Landing page loads without an exception.
- All five tabs open and their graphs display.
- March filter: 87 records and 765 hours.
- Extended duration filter: 63 records and 2,440 hours.
- Reset Filters restores the default 740 records and 5,124 hours.
- Simulator defaults: 1 backup person and 100% availability → 8-hour gap.
- Changing backup people to 2 and availability to 100% → 0-hour gap; changing availability to 50% → 8-hour gap.
- Reset Scenario restores all seven starting assumptions.
- Filtered CSV, ZIP analysis pack, and scenario CSV downloads succeed.
- Check layout on a desktop and a narrow/mobile browser.

## Troubleshooting

- **ModuleNotFoundError:** check that the build installed the package requirements.
- **FileNotFoundError:** confirm the `Data/Absenteeism_at_work.csv` path and filename casing are correct (Linux is case-sensitive).
- **Application failed to start:** check Render's build and runtime logs; verify the exact startup command and that `server = app.server` exists.
- **GitHub Actions tests failing:** run the same unittest command locally, then compare dependency versions in the logs.

Do not turn on Python/Dash debug mode on the public web service. The existing `if __name__ == "__main__"` guard prevents the local debug server from being used by Gunicorn.

Official deployment references: https://render.com/docs/deploy-flask and https://render.com/docs/web-services
