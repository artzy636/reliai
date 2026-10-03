import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forwards to the FastAPI backend (uvicorn api.main:app --port 8000)
      // so the frontend code always calls a same-origin /api path -- no
      // CORS to think about in the browser, no base-URL env var to keep
      // in sync between dev and demo.
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
