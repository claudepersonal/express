import { defineConfig } from 'vite';

// Client lives in ./client and builds to ./dist, which Express serves in production.
// `npm run dev` runs Vite on :5173 and proxies /api to the Express server on :8080.
export default defineConfig({
  root: 'client',
  build: { outDir: '../dist', emptyOutDir: true },
  server: { proxy: { '/api': 'http://localhost:8080', '/health': 'http://localhost:8080' } },
});
