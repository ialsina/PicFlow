# PicFlow

Interactive image labeling: keyboard classes (`1`–`9`, `0`), RGB histogram features (from the exploration notebook), PCA/UMAP cluster view, and swappable suggestion backends (kNN, sklearn MLP, PyTorch MLP).

## Prerequisites

- Python 3.10+
- Node.js 20+

## Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Restrict which folders may be used as image roots (comma-separated absolute paths).
# Leave unset to allow any path (local dev only).
export PICFLOW_ALLOWED_ROOTS=/path/to/your/photos

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
- Sessions: `POST /api/sessions`, `GET /api/sessions/{id}`, …

## Docker (optional)

Provide a `backend/Dockerfile` and `frontend/Dockerfile` or run locally as above. [`docker-compose.yml`](docker-compose.yml) expects an image volume at `PICFLOW_IMAGE_MOUNT`.
