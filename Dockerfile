# Match the environment the models were trained/serialized in (Python 3.13).
# The .pkl artifacts were saved with NumPy 2.4.6 (numpy._core); using an older
# Python/NumPy here causes "No module named 'numpy._core'" on load.
FROM python:3.13-slim

WORKDIR /app

# libgomp1 is the OpenMP runtime required by LightGBM at import/predict time.
# gcc/g++ cover any source builds for wheels lacking ARM/x86 prebuilds.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install inference-only deps (no ctgan / imbalanced-learn — training only).
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

# Appwrite injects PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# Run as non-root.
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
