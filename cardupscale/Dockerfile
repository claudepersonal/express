# CardCrop upscale tier: batch UI + API (server.py) over upscale_tier.py.
FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
# CPU-only torch wheel (no CUDA download).
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu -r requirements.txt
COPY upscale_tier.py server.py ./
COPY static ./static
# Weights are fetched from their release URLs and SHA-256 verified; a mismatch fails the build.
# The Real-ESRGAN ONNX graph is exported and checked against PyTorch (max diff < 1e-3).
RUN python upscale_tier.py --fetch-models && python upscale_tier.py --export-onnx \
    && pip uninstall -y onnx onnxscript
ENV CARDCROP_DATA_DIR=/tmp/cardcrop \
    PORT=8080
RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/cardcrop && chown -R app /tmp/cardcrop
USER app
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port ${PORT} --no-access-log"]
