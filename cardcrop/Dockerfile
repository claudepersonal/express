FROM node:22-bookworm-slim AS frontend
WORKDIR /web
COPY frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-osd libglib2.0-0 libstdc++6 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=frontend /usr/local/bin/node /usr/local/bin/node
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
COPY --from=frontend /web/dist /app/frontend/dist
RUN node --version && node --test tests/contracts.test.js tests/workbench.test.js
RUN python -m unittest discover -s tests -p 'test_fastapi.py' -v
ENV STORAGE_DIR=/data PYTHONUNBUFFERED=1 PORT=8080
EXPOSE 8080
CMD ["sh", "scripts/start.sh"]
