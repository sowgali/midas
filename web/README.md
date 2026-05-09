# midas web

React + Vite + TypeScript frontend for the **midas** cash-flow graph.

## Develop

```sh
cd web
npm install
npm run dev
# Then in another terminal: cd .. && uv run midas serve
# Open http://localhost:5173
```

The dev server proxies `/api/*` to `http://localhost:8000` (FastAPI), so the
frontend can call `fetch("/api/graph")` without CORS configuration.

## Build

```sh
npm run build      # tsc -b && vite build, emits dist/
npm run preview    # serve dist/ for a smoke test
```

## Layout

- `src/types.ts` — DTO mirrors of the FastAPI contracts.
- `src/api.ts` — typed `fetch` wrappers (all under `/api`).
- `src/main.tsx` — React entry, wraps `<QueryClientProvider>`.
- `src/App.tsx` — three-column layout: filters | graph | detail drawer.
- `src/components/GraphView.tsx` — `@xyflow/react` graph with `dagre` LR layout.
- `src/components/FilterPanel.tsx` — sector + as_of controls.
- `src/components/DetailDrawer.tsx` — entity / pair-deal inspector with
  evidence snippets and source links (provenance is the feature).
