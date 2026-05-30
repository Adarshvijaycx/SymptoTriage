# Render deployment image: full SymptoTriage app (FastAPI API + served frontend
# + SHAP). Python 3.11 to match the shap 0.46 / numpy 2.2 model artifacts.
# Render keeps the container warm, so loading the SHAP explainer at startup is
# fine (no serverless cold-start limit).
FROM python:3.11-slim

WORKDIR /app

# libgomp1 is the OpenMP runtime LightGBM needs; gcc/g++ cover source builds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt .
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy the whole project (compressed models in models/ are baked in).
COPY . .

# Full SHAP explainer is available on a warm host.
ENV LITE_EXPLAINER=0
ENV SERVE_FRONTEND=1
# Render injects $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
