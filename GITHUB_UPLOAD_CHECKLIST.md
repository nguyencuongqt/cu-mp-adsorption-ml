# GitHub Upload Checklist

This repository should contain only the minimum files needed for manuscript
review and reproducible reruns of the Python workflow.

## Upload

- `README.md` or this checklist as the repository landing note.
- `requirements.txt`, `Dockerfile`, `config.py`, and `main.py`.
- `src/analysis/*.py`, `src/rag/*.py`, and `src/ui/app.py`.
- `scripts/prepare_data.py`, `scripts/generate_legacy_outputs.py`,
  `scripts/make_figure5_llm_judge.py`, and
  `scripts/regenerate_llm_judge_figures.py`.
  Do not upload manuscript-editing helper scripts unless specifically needed
  for revision tracking.
- Curated numeric inputs in `data/`:
  - `raw_data_literature_master.xlsx`
  - `dataset.csv`
  - `field_data.csv`
  - `river_data.csv`
  - `test_cases.csv`
  - `real_river_reference.csv`
  - `modeled_river_reference.csv`
  - `paper_metadata.csv`
  - `paper_source_mapping.csv`
  - `ui_test_samples_20.csv`
- Optional lightweight result snapshots in `outputs/reports/*.csv` and
  `outputs/reports/*.txt`.
- `docs/DEPLOY.md` if deployment instructions are useful.

## Do Not Upload

- `.env`, `openai key.docx`, or any file containing API keys.
- `data/papers/` because these are full-text article PDFs.
- `data/chunks.jsonl` because it contains derived full-text chunks.
- `data/Metal adsorption MPs.xlsx` and reference `.xlsx` workbooks unless the
  journal explicitly asks for the broader private extraction workbook. The
  public raw/master file should be `data/raw_data_literature_master.xlsx`.
- `outputs/models/` and model/index artifacts such as `.joblib`, `.pkl`,
  `.gpickle`, and `.bin`.
- `tmp/`, `__pycache__/`, generated render folders, and local cache files.
- Manuscript/submission binaries such as `Manuscript Cu/`, TIFF figures,
  DOCX files, XLSX tables, and graphic abstract files unless the journal
  explicitly asks for them in the code repository.

## Suggested Data and Code Availability Text

The curated numeric datasets and Python scripts used in this study are
available in the project repository. The reproducible workflow includes data
preparation, missing-data diagnostics, model training, Quantile Random Forest
uncertainty estimation, SHAP analysis, field adjustment factor calibration,
river-scale estimation, figure/table regeneration, and the LLM Interpretation
System. Full-text article PDFs and derived text chunks are not redistributed
because of copyright restrictions; users can rebuild the retrieval index from
locally available PDFs using the provided ingestion scripts. API keys are not
included and must be provided locally through an environment variable or
project-local `.env` file.
