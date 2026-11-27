# Pothole Detection

## Run inference locally

Install the project dependencies and run inference on an image or video:

```powershell
python -m pip install -r requirements.txt
python demo.py --source data\example.jpg --device cpu
```

Inference results and per-frame telemetry are saved under `runs/detect`.

## Open the dashboard locally

```powershell
python -m streamlit run dashboard.py
```

The dashboard scans `runs/detect` for `situational_awareness.jsonl` files. A compact demo run is included and selected by default; choose a run, source, and frame in the sidebar.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository and choose `streamlit_dashboard/app.py` as the main file.
3. The app uses `streamlit_dashboard/requirements.txt` for its lightweight dashboard dependencies and `.streamlit/config.toml` for its theme.
4. The repository includes a compressed demo video and telemetry under `runs/detect/streamlit_demo`, which load automatically. Uploads remain optional for viewing other runs or annotated frames.

The situational-awareness score and related estimates are experimental and are not safety-certified driving controls.
