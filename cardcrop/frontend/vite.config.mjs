import { defineConfig } from 'vite';
const target=process.env.API_PROXY_URL || 'http://127.0.0.1:8000';
export default defineConfig({server:{proxy:Object.fromEntries(['/api','/uploads','/enhanced','/health'].map(path=>[path,{target,changeOrigin:true}]))},build:{outDir:'dist',emptyOutDir:true}});
