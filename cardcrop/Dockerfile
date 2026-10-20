FROM node:22-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd libgl1 libglib2.0-0 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir 'ultralytics>=8.3,<9'
COPY . .
COPY --from=frontend /web/dist /app/frontend/dist
RUN mkdir -p /app/models /data/uploads /data/enhanced
# Preserve the verified detector provisioning from the current GitHub backend.
ARG CARD_DETECTOR_URL=https://raw.githubusercontent.com/claudepersonal/CardEnhance/2521c65085e3640d4f16a194db991305637e64ac/models/card-seg/v1/best.pt
ARG CARD_DETECTOR_SHA256=3cd5708b94738d1ba24cdbba73db6b8dbb49d31c903b3c2c69346c87f669a1d6
RUN python -c "import hashlib,sys,urllib.request; data=urllib.request.urlopen(sys.argv[1],timeout=120).read(); digest=hashlib.sha256(data).hexdigest(); assert digest==sys.argv[2], 'card detector checksum mismatch: '+digest; open('/app/models/card_detector.pt','wb').write(data)" "$CARD_DETECTOR_URL" "$CARD_DETECTOR_SHA256"
RUN node --version && node --test tests/contracts.test.js tests/workbench.test.js
RUN python -m unittest discover -s tests -p 'test_fastapi.py' -v
ENV STORAGE_DIR=/data PYTHONUNBUFFERED=1 PORT=8080 YOLO_MODEL=/app/models/card_detector.pt YOLO_CONFIG_DIR=/tmp/ultralytics
EXPOSE 8080
CMD ["sh", "scripts/start.sh"]
