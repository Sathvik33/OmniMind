# Aegis Frontend

React 19 + Vite SPA for the multimodal RAG workspace.

## Run

```bash
npm install
cp .env.example .env   # VITE_BACKEND_URL=http://127.0.0.1:8000
npm run dev            # http://localhost:5173
```

```bash
npm run build
npm run preview
```

## Auth UI

Landing → Log in / Sign up (email + password). JWT in `localStorage`. Sidebar lists past chats; each chat loads its own messages and uploads. Queries send `session_id` only — the server resolves that chat’s artifact scope.

## What you get

| Surface | Role |
|---------|------|
| `Landing` | Brand-first entry (paper / desk visual) |
| `Workspace` | Chat state, upload → job poll, scoped `artifact_ids` |
| `MessageList` | Text bubbles + **in-thread document cards** + stream caret |
| `Composer` | Attach tray + send (locked until embeddings ready) |
| `api/client.ts` | `uploadFile`, `getJobStatus`, `streamQuery`, `cleanStreamText` |

## Streaming contract

1. `POST {VITE_BACKEND_URL}/query-stream` with `{ query, artifact_ids }`
2. Read `ReadableStream` as UTF-8 text chunks
3. Strip `[Retrieved:…]` / confidence / warning lines for display
4. Show caret on the last assistant message while `streaming === true`

## Upload contract

1. Document card inserted immediately (`uploading`)
2. `POST /upload` → `job_id` / `artifact_id`
3. Poll `GET /jobs/{id}` every ~1.5s
4. Card → `ready` · unlock ask · keep `artifact_id` in scope

## Design tokens

Defined in `src/index.css` — forest ink on warm paper (`Fraunces` + `Figtree`). Avoid neon “AI SaaS” chrome; keep the desk metaphor.

## Stack

- React 19 + TypeScript
- Vite 8
- Framer Motion (page / message enter)
- Plain CSS modules co-located with components

Legacy Streamlit UI lives under `legacy-streamlit-app/` and is not the primary surface.
