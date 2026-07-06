# Deployment Guide

## Local run

```powershell
pip install -r requirements.txt
python main.py --skip-ui
streamlit run src/ui/app.py
```

## Docker run

```bash
docker build -t cu-mp-predictor .
docker run -p 8501:8501 \
  -e OPENAI_API_KEY=... \
  -v $(pwd)/data:/app/data \
  cu-mp-predictor
```

## Streamlit Cloud deploy

1. Push to GitHub.
2. Connect the repository at share.streamlit.io.
3. Set `OPENAI_API_KEY` in Secrets if using LLM-enabled modes.
4. Set the main file to `src/ui/app.py`.
