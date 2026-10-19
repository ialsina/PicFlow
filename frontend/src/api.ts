const BASE = '/api'

function errorMessageFromBody(text: string, fallback: string): string {
  if (!text) return fallback
  try {
    const body = JSON.parse(text) as { detail?: unknown }
    const detail = body.detail
    if (typeof detail === 'string') return detail
    if (Array.isArray(detail)) {
      return detail.map((d) => (typeof d === 'object' && d && 'msg' in d ? String(d.msg) : String(d))).join('; ')
    }
  } catch {
    /* plain text */
  }
  return text
}

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  const text = await res.text()
  if (!res.ok) {
    throw new Error(errorMessageFromBody(text, res.statusText))
  }
  if (res.status === 204) {
    return undefined as T
  }
  return JSON.parse(text) as T
}

export type SessionSummary = {
  id: string
  image_root: string
  config: Record<string, unknown>
  index_status: string
  index_progress: number
  index_message: string
  total_images: number
  labeled_count: number
  unlabeled_count: number
}

export type SessionStatus = {
  index_status: string
  index_progress: number
  index_message: string
  training_status: string
  training_message: string
  projection_status: string
  projection_message: string
  projection_method: string | null
}

export type ImageDetail = {
  id: number
  idx: number
  path: string
  filename: string
  label: number | null
  label_source: string | null
}

export type Suggestion = {
  ready: boolean
  label?: number | null
  confidence?: number | null
  provider?: string | null
  reason?: string | null
}

export type ProjectionPoint = {
  id: number
  x: number
  y: number
  label: number | null
  idx: number
  kmeans_cluster?: number | null
}

export type Projection = {
  points: ProjectionPoint[]
  method: 'pca' | 'umap'
  explained_variance?: number[] | null
  pending?: boolean
}

export type Stats = {
  total: number
  labeled: number
  unlabeled: number
  per_label: Record<string, number>
  model_ready: boolean
  provider: string | null
  last_train_ms: number | null
  holdout_accuracy: number | null
  training_status: string
  projection_method: string | null
}

export type ProviderInfo = {
  id: string
  name: string
  config_schema: Record<string, unknown>
}

export function createSession(imageRoot: string, config?: Record<string, unknown>) {
  return json<SessionSummary>(`${BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image_root: imageRoot, config }),
  })
}

export function listSessions(limit = 50) {
  return json<{ items: SessionSummary[] }>(`${BASE}/sessions?limit=${limit}`)
}

export function deleteSession(sessionId: string) {
  return json<void>(`${BASE}/sessions/${sessionId}`, { method: 'DELETE' })
}

export function getSession(sessionId: string) {
  return json<SessionSummary>(`${BASE}/sessions/${sessionId}`)
}

export function getStatus(sessionId: string) {
  return json<SessionStatus>(`${BASE}/sessions/${sessionId}/status`)
}

export function getImageIds(sessionId: string, unlabeledOnly = false) {
  return json<{ items: { id: number; idx: number; label: number | null }[] }>(
    `${BASE}/sessions/${sessionId}/image-ids?unlabeled_only=${unlabeledOnly}`,
  )
}

export function getImage(sessionId: string, imageId: number) {
  return json<ImageDetail>(`${BASE}/sessions/${sessionId}/images/${imageId}`)
}

export function imageFileUrl(sessionId: string, imageId: number, thumb?: number) {
  const q = thumb ? `?thumb=${thumb}` : ''
  return `${BASE}/sessions/${sessionId}/images/${imageId}/file${q}`
}

export function setLabel(
  sessionId: string,
  imageId: number,
  label: number | null,
  source: 'user' | 'accepted_suggestion' = 'user',
) {
  return json<ImageDetail>(`${BASE}/sessions/${sessionId}/images/${imageId}/label`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ label, source }),
  })
}

export function getSuggestion(sessionId: string, imageId: number) {
  return json<Suggestion>(`${BASE}/sessions/${sessionId}/images/${imageId}/suggestion`)
}

export function getProjection(sessionId: string, method?: 'pca' | 'umap') {
  const q = method ? `?method=${method}` : ''
  return json<Projection>(`${BASE}/sessions/${sessionId}/projection${q}`)
}

export function getStats(sessionId: string) {
  return json<Stats>(`${BASE}/sessions/${sessionId}/stats`)
}

export function trainSession(sessionId: string) {
  return json<{ success: boolean; message: string }>(`${BASE}/sessions/${sessionId}/train`, {
    method: 'POST',
  })
}

export function patchConfig(sessionId: string, config: Record<string, unknown>) {
  return json<{ config: Record<string, unknown> }>(`${BASE}/sessions/${sessionId}/config`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ config }),
  })
}

export function listProviders() {
  return json<ProviderInfo[]>(`${BASE}/providers`)
}

export type LabelImportResult = {
  applied: number
  skipped_unknown: number
  skipped_ambiguous: number
  skipped_invalid: number
  errors: string[]
  config_updated: boolean
}

export async function exportLabels(sessionId: string): Promise<void> {
  const res = await fetch(`${BASE}/sessions/${sessionId}/labels/export`)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(errorMessageFromBody(text, res.statusText))
  }
  const blob = await res.blob()
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `picflow-${sessionId}-labels.zip`
  link.click()
  URL.revokeObjectURL(url)
}

export function importLabels(sessionId: string, formData: FormData) {
  return json<LabelImportResult>(`${BASE}/sessions/${sessionId}/labels/import`, {
    method: 'POST',
    body: formData,
  })
}
