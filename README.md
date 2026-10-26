# Cu adsorption on microplastics: laboratory isotherms, field scaling and LLM-assisted explanation

Code, curated data and result tables for the manuscript *From laboratory isotherms to rivers: machine learning prediction of
copper adsorption on microplastics, field scaling and LLM-assisted explanation* (Environmental Pollution, revision,
ENVPOL-D-26-10175), and the web application described in Section 3.2.3.

**Web application:** `streamlit run src/ui/app.py` (local), or the hosted version: *[add the Streamlit Community Cloud URL here]*.

## Contents

| Path | Content |
|---|---|
| `data/raw_data_literature_master.xlsx` | Curated literature extraction: laboratory isotherms, sources, data corrections |
| `data/dataset_grouped.csv` | 961 laboratory observations from 32 studies with study and isotherm identifiers (used by all revision analyses and the app) |
| `data/field_data.csv`, `data/field_evidence_tiers.csv` | Field records: Cu on riverine microplastics with paired water Cu, and supporting records |
| `data/river_data.csv`, `data/river_input_sources.csv`, `data/global_*.csv` | River and global inputs of the scale-up |
| `data/expert_scores_by_item.csv` | Blinded expert scores (raters LVT and PB) of the 60 explanations of 15 cases, with case and condition |
| `data/templates/` | Extraction templates for new laboratory, field and river data |
| `src/analysis/` | Model training, study-grouped validation, conformal intervals, interpretation, field-to-laboratory ratio, river and global estimates |
| `src/analysis/predictor.py` | The model used by the application (pooled Freundlich isotherm with polymer, ageing status and adsorption system) |
| `src/rag/` | LLM explanation system (C1 rule-based; C2 retrieval; C3 retrieval + knowledge graph), evaluation and LLM judges |
| `src/ui/app.py` | Streamlit application (`src/ui/app_submitted_version.py`: interface of the first submission) |
| `scripts/` | Data curation steps, figures and tables of the manuscript, expert-score analysis |
| `outputs/reports/` | Result tables behind every number, table and figure of the manuscript and SI |
| `outputs/models/corpus_v2_papers.csv` | The 41 source articles of the retrieval corpus (keys and DOIs) |

Not included: full-text articles (`data/papers/`), the text chunks and retrieval index derived from them
(`data/chunks.jsonl`, `outputs/models/faiss_index.bin`, `chunks_metadata.pkl`), trained model binaries, manuscript files and API
keys. The articles cannot be redistributed; users with access can place the PDFs in `data/papers/` and rebuild the index
(`python src/rag/build_corpus_v2.py`). Scripts that read from `Manuscript/` (expert scoring sheets, figure output folders) need
the corresponding local files or create the folders.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt   # full workflow (Python 3.12; results produced with Python 3.12.10)
```

The application alone needs only `pip install -r src/ui/requirements.txt`.

For the LLM modes C2 and C3 and the LLM judges, copy `.env.example` to `.env` and add `OPENAI_API_KEY`
(generation, GPT-4o mini) and `ANTHROPIC_API_KEY` (independent judge, Claude Haiku 4.5). Never commit `.env`.

## Reproducing the revision analyses

```bash
python scripts/build_grouped_dataset.py            # data/dataset_grouped.csv and QC flags
python src/analysis/grouped_validation.py          # random vs study-grouped validation of all models
python scripts/table1_model_performance.py         # Table 1 and the low-concentration hold-out
python src/analysis/final_model_validation.py      # nested leave-one-study-out and conformal prediction intervals
python src/analysis/transfer_analysis.py           # transfer to unseen studies; one local measurement
python src/analysis/low_ce_extrapolation.py        # behaviour at river concentrations
python src/analysis/interpretation_main_model.py   # SHAP and response curves (Fig. 3)
python src/analysis/faf_scaling.py                 # field-to-laboratory ratio
python src/analysis/global_flux_constrained.py     # global estimate
python src/analysis/river_application.py           # rivers with measured inputs (Table 3)
python src/analysis/global_monte_carlo.py          # Monte Carlo and sensitivity
python src/rag/llm_eval_v2.py                      # LLM evaluation, 30 cases x 4 conditions (needs API keys and the index)
python scripts/llm_eval_v2_analysis.py             # summaries of the LLM evaluation
python scripts/expert_scoring_analysis.py          # expert scores, agreement (ICC) and differences between conditions
python scripts/make_revised_figures.py             # Figs. 2-5 from outputs/reports
```

Each script documents its inputs and outputs in its header. The result tables used in the manuscript are in
`outputs/reports/`, so the figures can be redrawn without refitting the models.

## Web application

```bash
streamlit run src/ui/app.py
```

Panel E shows the 120 explanations evaluated in the manuscript (30 cases × C0–C3) with their source articles, the checks
without a judge and the scores of the LLM judges, so the C2 and C3 outputs can be inspected without an API key.

Inputs: polymer, ageing status, adsorption system and equilibrium Cu concentration (Ce); optionally one measured (Ce, qe)
pair for the material, which narrows the 90% interval from ×/÷29 to ×/÷2.6. Outputs: laboratory-equilibrium qe and its
interval, the field-scale value (laboratory qe × field-to-laboratory ratio 0.57) with a 90% interval that also includes the
variation of the ratio between rivers, the contribution of each input, validity warnings, an explanation (C1 without LLM;
C2/C3 with the retrieval index and an API key) and reports in .txt and .docx. Batch prediction from CSV/XLSX files with the
columns `ReT, AgS, AdC, Ce` (optional `anchor_Ce, anchor_qe`). Deployment: `docs/DEPLOY.md`.

## Licence

Code: MIT (`LICENSE`). Data in `data/` and result tables in `outputs/`: CC BY 4.0.

## Citation

Please cite the article (details to be added on publication) and the archived version of this repository
(Zenodo DOI: *[add after the first release]*). `CITATION.cff` gives the citation in machine-readable form.
