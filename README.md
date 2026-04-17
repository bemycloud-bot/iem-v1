# IEM Prediction Web App (Rebuilt)

This web app is rebuilt from scratch for:
- Model: `best_original_intensive_v1_grid_lr.joblib`
- Notebook reference: `Top6IEM_01.4.10 [Remake](final).ipynb`

## Features
- Upload CSV
- Click **Run Prediction** to infer disease probabilities
- Show top-1/top-2/top-3 disease with probability
- Show marker-level **MoM** and **cut-off** values
- Auto-send formal Discord reports (summary + PNG report + additional pattern PNG + 3D graph PNG)

## Setup

```bash
cd "webapp_model_inference"
/opt/anaconda3/bin/python3.12 -m venv ../.venv-webapp312
source ../.venv-webapp312/bin/activate
pip install -r requirements.txt
```

## Run

```bash
cd "webapp_model_inference"
source ../.venv-webapp312/bin/activate
streamlit run streamlit_app.py --server.port 8515
```

Open: `http://localhost:8515`

## Default assets
- Model: `comparisons/best_original_intensive_v1_grid_lr.joblib`
- Class mapping: `comparisons/class_mapping.csv`
- Disease patterns: `disease_patterns.json`
- Cut-off Excel: `Edit20260116_KCMH_cut-off-Summary_N28778_48to72_Normals.xlsx`
- Training CSV (optional, for marker weighting only): `new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv`

## Optional Discord webhook
Create `.env` in this folder:

```env
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxxx/yyyy
MODEL_PATH=comparisons/best_original_intensive_v1_grid_lr.joblib
CLASS_MAPPING_PATH=comparisons/class_mapping.csv
TRAIN_CSV_PATH=new2Top6EditRename_New_Clean_Chula_MPIEM_group.csv
```
