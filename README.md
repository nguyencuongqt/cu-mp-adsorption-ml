# Cu-MP Adsorption Prediction

Python workflow for predicting Cu2+ adsorption by microplastics, estimating
Quantile Random Forest uncertainty, calibrating river-scale predictions with a
field adjustment factor, and generating SHAP and LLM-assisted interpretation
outputs for manuscript review.

## Repository Contents

- `main.py` - end-to-end resumable workflow.
- `config.py` - project paths, model settings, feature columns, and constants.
- `src/analysis/` - missing-data diagnostics, model training, QRF uncertainty,
  SHAP analysis, FAF calibration, pH sensitivity, and river-scale estimation.
- `src/rag/` - document ingestion, vector retrieval, knowledge graph, LLM
  interpretation, and ablation evaluation.
- `src/ui/app.py` - Streamlit application.
- `scripts/` - data preparation and manuscript-style figure/table generation.
- `data/raw_data_literature_master.xlsx` - raw curated literature data,
  model-ready Cu subset, source metadata, and validation summary.
- `data/*.csv` - curated numeric datasets used by the workflow.
- `outputs/reports/` - optional lightweight result snapshots.

Full-text PDFs, generated text chunks, trained model binaries, and API keys are
not included in the public repository.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For LLM-enabled modes, set `OPENAI_API_KEY` in your environment or in a local
`.env` file. Do not commit `.env`.

## Reproduce Core Outputs

```bash
python main.py --skip-llm --skip-ui
python scripts/generate_legacy_outputs.py
```

To rebuild the LLM retrieval artifacts, place locally accessible article PDFs in
`data/papers/`, then run:

```bash
python main.py --from 10 --skip-ui
```

To launch the app:

```bash
streamlit run src/ui/app.py
```
