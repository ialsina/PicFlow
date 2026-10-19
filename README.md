# PicFlow

Interactive image labeling: keyboard classes (`1`–`9`, `0`), RGB histogram features (from the exploration notebook), PCA/UMAP cluster view, and swappable suggestion backends (kNN, sklearn MLP, PyTorch MLP).

The **platform API** (`backend/`) handles sessions, Postgres, local image paths, and histogram feature extraction. **ML training, suggestions, and projections** run in a separate **ml-service** (`ml-service/`), called over HTTP with feature vectors only (images never leave the platform).

## Prerequisites

- Python 3.12+
- Node.js 20+
- PostgreSQL 16+ (or use Docker Compose below)

## Architecture

| Component | Role |
|-----------|------|
| `backend` | FastAPI, Postgres, image bind/serve, histogram features |
| `ml-service` | kNN / MLP providers, PCA/UMAP/k-means on vectors |
| `frontend` | Vite UI |

Switch **fully local** vs **distributed** (ML on a remote VPS) with **runtime** env vars — same Docker images, no rebuild. See [Deployment modes](#deployment-modes).

## Backend (platform API)

Start PostgreSQL (example with Docker):

```bash
docker compose up -d postgres
```

Start **ml-service** (required for suggestions and scatter plot):

```bash
cd ml-service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

Optional PyTorch provider in ml-service:

```bash
pip install torch
```

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export PICFLOW_DATABASE_URL=postgresql+psycopg://picflow:picflow@localhost:5432/picflow
export PICFLOW_ALLOWED_ROOTS=/path/to/your/photos
export PICFLOW_ML_BASE_URL=http://127.0.0.1:8001
export PICFLOW_ML_API_KEY=dev-local-key

alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

See [`backend/.env.example`](backend/.env.example).

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

- Health: `GET /api/health` (includes `ml_reachable` when ML is enabled)
- Providers: `GET /api/providers` (proxied from ml-service)
- Sessions: `GET /api/sessions`, `POST /api/sessions`, `GET /api/sessions/{id}`, `DELETE /api/sessions/{id}`, …

### Label export / import

Export labeled images for a session (Settings → **Labels (CSV)** in the UI, or API):

- `GET /api/sessions/{id}/labels/export` — ZIP named `picflow-{session_id}-labels.zip` containing:
  - **`picflow-{session_id}-labels.csv`**: `image_path` (relative to session `image_root`), `image_name` (basename), `label` (0–9). Only labeled images are included.
  - **`picflow-{session_id}-label_names.csv`**: `label`, `name` for class ids used in the export (plus any custom names from session config).

Import labels into an existing session (images must already be indexed):

- `POST /api/sessions/{id}/labels/import` — `multipart/form-data` with either:
  - `archive` — ZIP with those CSV files (or legacy `labels.csv` / `label_names.csv` names), or
  - `labels` — labels CSV (required), optional `label_names` — label names CSV

Matching prefers `image_path`, then unique `image_name`. Existing labels are overwritten. The response reports counts (`applied`, `skipped_unknown`, `skipped_ambiguous`, `skipped_invalid`) and sample `errors`.

## Docker (optional)

### Fully local (platform + ML in Compose)

```bash
cp .env.example .env   # optional; defaults work for local ML
PICFLOW_IMAGE_MOUNT=/path/to/photos docker compose --profile ml up --build
```

Open http://localhost:5173. Backend talks to `http://ml-service:8001` on the internal network.

### Distributed (ML on a remote VPS)

1. Deploy **only** `ml-service` on the VPS (build from `ml-service/`, set `PICFLOW_ML_API_KEY`, put TLS in front of port 8001).
2. On your labeling machine, run compose **without** the `ml` profile and point the backend at the VPS:

```bash
export PICFLOW_ML_BASE_URL=https://ml.your-vps.example
export PICFLOW_ML_API_KEY=<shared-secret>
PICFLOW_IMAGE_MOUNT=/path/to/photos docker compose up --build
```

The backend image is unchanged; only env vars differ.

### Deployment modes

| Variable | Local Compose | Remote VPS |
|----------|---------------|------------|
| `PICFLOW_ML_BASE_URL` | `http://ml-service:8001` (default) | `https://ml.your-vps.example` |
| `PICFLOW_ML_API_KEY` | `dev-local-key` (match ml-service) | strong shared secret |
| Compose | `docker compose --profile ml up` | `docker compose up` (no ml profile) |
| `PICFLOW_ML_ENABLED` | `true` (default) | `false` only if you want labeling without ML |

See [`.env.example`](.env.example).

If host port 5432 is already in use:

```bash
PICFLOW_POSTGRES_PORT=5434 PICFLOW_IMAGE_MOUNT=/path/to/photos docker compose --profile ml up --build -d
```

Do **not** set `PICFLOW_DATABASE_URL` to `...@postgres:5434` for Compose: the backend container must use `postgres:5432` (internal port). Use `localhost:5434` only when the API runs **on the host** (e.g. `uvicorn`), not inside Docker.

### Troubleshooting “Database unavailable”

1. **Only one API on port 8000.** If `uvicorn` is running locally (IDE or terminal), stop it before using the Docker backend, or you may hit the wrong process.
2. **Check the API you are using:** `curl -s http://localhost:8000/api/health` should return `{"status":"ok","ml_reachable":...}`.
3. **ML unavailable:** ensure ml-service is running (`--profile ml`) or `PICFLOW_ML_BASE_URL` reaches your VPS; `curl -s http://localhost:8001/v1/health` when running ML locally on the host.
4. **Inside Docker (should work with compose as-is):**
   `docker compose exec backend python -c "from sqlalchemy import text; from app.db import engine; print(engine.connect().execute(text('SELECT 1')).scalar())"`
5. **If the error mentions host `postgres` but you run uvicorn on the host:** that URL is for containers only — use `localhost` and your mapped port (`5434` when `PICFLOW_POSTGRES_PORT=5434`). See `backend/.env.example`.
6. **Reset stack:** `docker compose down && docker compose --profile ml up --build -d` then wait until `docker compose ps` shows services **healthy**.

When the backend is configured with a single allowed root (as in Compose), the UI picks it up from `GET /api/config` and starts sessions without asking for a path. Otherwise use an absolute folder path on the machine where the API runs. In Docker that is typically **`/data/images`** (or a subfolder) — inside the backend container, not on the host.

[`docker-compose.yml`](docker-compose.yml) runs PostgreSQL, the platform API, optional **ml-service** (profile `ml`), and the frontend. The backend uses `PICFLOW_DATABASE_URL` to reach the `postgres` service and applies Alembic migrations on startup. `API_PROXY_TARGET=http://backend:8000` lets the Vite dev server in the frontend container reach the API. Local `npm run dev` still proxies to `127.0.0.1:8000` by default.
