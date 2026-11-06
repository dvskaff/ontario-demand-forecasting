# The forecast API: FastAPI + onnxruntime serving the champion model. No LightGBM, MLflow or training code inside.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app MODEL_DIR=/app/model
WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The feature code is shared with training, so the API builds features exactly the way the model was trained.
COPY src/demand/__init__.py src/demand/features.py demand/
COPY serving/app app/
COPY serving/model model/

RUN useradd --create-home --uid 10001 api && chown -R api /app
USER api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

LABEL org.opencontainers.image.source="https://github.com/dvskaff/ontario-demand-forecasting" \
      org.opencontainers.image.description="Next-day Ontario electricity demand forecast API (FastAPI + ONNX Runtime)" \
      org.opencontainers.image.licenses="MIT"
