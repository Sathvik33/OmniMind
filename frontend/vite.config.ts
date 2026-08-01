import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND = 'http://127.0.0.1:8000'

function apiProxy(extra: Record<string, unknown> = {}) {
  return {
    target: BACKEND,
    changeOrigin: true,
    secure: false,
    // Long RAG + local LLM — do not cut the tunnel mid-stream
    timeout: 0,
    proxyTimeout: 0,
    ...extra,
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    allowedHosts: true,
    proxy: {
      // Longer path first — avoid /query swallowing /query-stream
      '/query-stream': apiProxy(),
      '/query': apiProxy(),
      '/auth': apiProxy(),
      '/chats': apiProxy(),
      '/upload': apiProxy(),
      '/jobs': apiProxy(),
      '/health': apiProxy(),
      '/evaluate': apiProxy(),
      '/monitor': apiProxy(),
      '/feedback': apiProxy(),
      '/docs': apiProxy(),
      '/openapi.json': apiProxy(),
    },
  },
})
