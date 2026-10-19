# PicFlow

Interactive image labeling: keyboard classes (`1`–`9`, `0`), RGB histogram features (from the exploration notebook), PCA/UMAP cluster view, and swappable suggestion backends (kNN, sklearn MLP, PyTorch MLP).

## Prerequisites

- Python 3.12+
- Node.js 20+
- PostgreSQL 16+ (or use Docker Compose below)

## Backend

Start PostgreSQL (example with Docker):

```bash
docker compose up -d postgres
```

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Default: postgresql+psycopg://picflow:picflow@localhost:5432/picflow
# If compose maps Postgres to another host port (PICFLOW_POSTGRES_PORT), match it here:
export PICFLOW_DATABASE_URL=postgresql+psycopg://picflow:picflow@localhost:5432/picflow

# Restrict which folders may be used as image roots (comma-separated absolute paths).
# Leave unset to allow any path (local dev only).
export PICFLOW_ALLOWED_ROOTS=/path/to/your/photos

# Apply schema migrations (also runs automatically on API startup)
alembic upgrade head

uvicorn app.main:app --reload --port 8000
```

Optional PyTorch provider:

```bash
pip install torch
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 — the dev server proxies `/api` to the backend.

## Usage

1. Enter the **absolute path** to a folder of RGB images on the machine running the backend.
2. Wait for indexing to finish (histogram features are computed in the background).
3. Label with number keys; `←`/`→` or `j`/`k` to navigate; `Backspace` to clear.
4. When enough labels exist (see Settings), the suggestion bar appears; **Tab** or **Enter** accepts.
5. Right panel: PCA scatter (UMAP via Settings → Recompute UMAP), per-class counts, model status.

## Configuration

See [`config.example.yaml`](config.example.yaml). Most values can be changed in the UI Settings drawer or via `PATCH /api/sessions/{id}/config`.

## API

- Health: `GET /api/health`
- Providers: `GET /api/providers`
- Sessions: `GET /api/sessions`, `POST /api/sessions`, `GET /api/sessions/{id}`, `DELETE /api/sessions/{id}`, …

### Label export / import

Export labeled images for a session (Settings → **Labels (CSV)** in the UI, or API):

- `GET /api/sessions/{id}/labels/export` — ZIP containing:
  - **`labels.csv`**: `image_path` (relative to session `image_root`), `image_name` (basename), `label` (0–9). Only labeled images are included.
  - **`label_names.csv`**: `label`, `name` for class ids used in the export (plus any custom names from session config).

Import labels into an existing session (images must already be indexed):

- `POST /api/sessions/{id}/labels/import` — `multipart/form-data` with either:
  - `archive` — ZIP with the CSV files above, or
  - `labels` — `labels.csv` (required), optional `label_names` — `label_names.csv`

Matching prefers `image_path`, then unique `image_name`. Existing labels are overwritten. The response reports counts (`applied`, `skipped_unknown`, `skipped_ambiguous`, `skipped_invalid`) and sample `errors`.

## Docker (optional)

```bash
# Mount your image folder (host path) and open http://localhost:5173
PICFLOW_IMAGE_MOUNT=/path/to/photos docker compose up --build
```

If host port 5432 is already in use, map Postgres to another host port only:

```bash
PICFLOW_POSTGRES_PORT=5434 PICFLOW_IMAGE_MOUNT=/path/to/photos docker compose up --build -d
```

Do **not** set `PICFLOW_DATABASE_URL` to `...@postgres:5434` for Compose: the backend container must use `postgres:5432` (internal port). Use `localhost:5434` only when the API runs **on the host** (e.g. `uvicorn`), not inside Docker.

### Troubleshooting “Database unavailable”

1. **Only one API on port 8000.** If `uvicorn` is running locally (IDE or terminal), stop it before using the Docker backend, or you may hit the wrong process.
2. **Check the API you are using:** `curl -s http://localhost:8000/api/health` should return `{"status":"ok"}`.
3. **Inside Docker (should work with compose as-is):**
   `docker compose exec backend python -c "from sqlalchemy import text; from app.db import engine; print(engine.connect().execute(text('SELECT 1')).scalar())"`
4. **If the error mentions host `postgres` but you run uvicorn on the host:** that URL is for containers only — use `localhost` and your mapped port (`5434` when `PICFLOW_POSTGRES_PORT=5434`). See `backend/.env.example`.
5. **Reset stack:** `docker compose down && docker compose up --build -d` then wait until `docker compose ps` shows backend and postgres **healthy**.

Use **`/data/images`** (or a subfolder) as the image root in the UI — that path is inside the backend container, not on the host.

[`docker-compose.yml`](docker-compose.yml) runs PostgreSQL, the API, and the frontend. The backend uses `PICFLOW_DATABASE_URL` to reach the `postgres` service and applies Alembic migrations on startup. `API_PROXY_TARGET=http://backend:8000` lets the Vite dev server in the frontend container reach the API. Local `npm run dev` still proxies to `127.0.0.1:8000` by default.
