# Deploying the web application

## Local

```bash
pip install -r src/ui/requirements.txt      # application only; requirements.txt for the full workflow
streamlit run src/ui/app.py
```

C2 and C3 additionally need `pip install openai sentence-transformers faiss-cpu`, the retrieval index built from the source
articles (`python src/rag/build_corpus_v2.py`, with the PDFs in `data/papers/`) and `OPENAI_API_KEY` in `.env`.

## Streamlit Community Cloud (free)

1. Push this repository to GitHub (public).
2. Sign in at https://share.streamlit.io with the GitHub account and choose **Create app → Deploy a public app from GitHub**.
3. Repository `nguyencuongqt/cu-mp-adsorption-ml`, branch `main`, main file path `src/ui/app.py`.
4. Optional: choose a short App URL (e.g. `cu-mp-adsorption`), giving `https://cu-mp-adsorption.streamlit.app`.
5. Advanced settings: Python 3.12. No secrets are needed.
6. Deploy. The first build takes a few minutes; later pushes to `main` redeploy automatically.

Community Cloud installs `src/ui/requirements.txt` (the dependency file next to the entry point takes precedence over the
root file), so the hosted app does not install the machine-learning and retrieval libraries.

The hosted app offers C1 (rule-based explanation, no LLM). C2 and C3 are disabled there because the retrieval index is
built from full-text articles that cannot be redistributed; the app says so and points to the local installation. Panel E
shows the 120 explanations evaluated in the manuscript (C0-C3, with checks and judge scores), so the C2 and C3 outputs can
be inspected online. Do not
add an OpenAI key to a public deployment: every visitor would use it.

Apps on Community Cloud sleep after a period without visits; the first visitor after that waits about a minute.

## Docker

```bash
docker build -t cu-mp-predictor .
docker run -p 8501:8501 cu-mp-predictor
```
