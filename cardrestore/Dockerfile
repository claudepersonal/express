# ── 1. Build the Vite client ──
FROM node:20-bookworm-slim AS client
WORKDIR /build
COPY package.json ./
RUN npm install --no-audit --no-fund
COPY vite.config.mjs ./
COPY client ./client
RUN npm run build

# ── 2. Runtime: Express + Python Real-ESRGAN worker (CPU torch) ──
FROM node:20-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip tesseract-ocr tesseract-ocr-eng ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch wheels (no CUDA download); spandrel loads RealESRGAN_x4plus.
RUN pip3 install --no-cache-dir --break-system-packages \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.5.1+cpu spandrel==0.4.2 \
      "opencv-python-headless>=4.9,<4.13" "numpy>=1.26,<2.3" "Pillow>=10" pytesseract

WORKDIR /app

# Real-ESRGAN x4plus weights, fetched at build time and verified by SHA-256.
# The worker re-checks the checksum before loading; a mismatch fails loudly.
ARG MODEL_URL=https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
ARG MODEL_SHA256=4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1
RUN mkdir -p models && python3 -c "import hashlib,sys,urllib.request; d=urllib.request.urlopen(sys.argv[1],timeout=300).read(); h=hashlib.sha256(d).hexdigest(); assert h==sys.argv[2], 'model checksum mismatch: '+h; open('models/RealESRGAN_x4plus.pth','wb').write(d); print('model:',len(d),'bytes',h)" "$MODEL_URL" "$MODEL_SHA256"

COPY package.json ./
RUN npm install --omit=dev --no-audit --no-fund
COPY server ./server
COPY worker ./worker
COPY --from=client /build/dist ./dist

ENV NODE_ENV=production \
    STORAGE_DIR=/tmp/cardrestore \
    RESTORE_MODEL=/app/models/RealESRGAN_x4plus.pth \
    PORT=8080
RUN useradd --create-home --uid 10001 app && mkdir -p /tmp/cardrestore && chown -R app /tmp/cardrestore
USER app
EXPOSE 8080
CMD ["node", "server/index.js"]
