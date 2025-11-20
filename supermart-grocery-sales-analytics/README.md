# Supermart Grocery Sales – Retail Analytics & ML Project

This repository contains an end-to-end data analysis and simple ML pipeline for the
**Supermart Grocery Sales** dataset.

**Dataset included:** `data/Supermart Grocery Sales - Retail Analytics Dataset.csv`

## What's inside

- `data/` – contains the dataset (CSV).
- `notebooks/` – exploratory notebook.
- `src/` – Python scripts: `eda.py`, `preprocess.py`, `model.py`, `utils.py`.
- `visuals/` – generated plots (gitignored).
- `requirements.txt` – Python dependencies.
- `.gitignore`

## Quick start

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate    # Windows PowerShell
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run EDA and generate visuals:
```bash
python src/eda.py
```

4. Preprocess data:
```bash
python src/preprocess.py
```

5. Train model:
```bash
python src/model.py
```

All generated plots will be saved to `visuals/`.

## Author

Created for Kushagra Gupta.