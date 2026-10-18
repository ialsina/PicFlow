import { useEffect, useState, type FormEvent } from 'react'
import { listSessions, type SessionSummary } from '../api'

type Props = {
  onStart: (imageRoot: string) => void
  onRecover: (sessionId: string) => void
  onRemove: (sessionId: string) => void | Promise<void>
  listKey: number
  loading: boolean
  error: string | null
}

function truncatePath(path: string, max = 56) {
  if (path.length <= max) return path
  const head = Math.floor((max - 1) / 2)
  const tail = max - 1 - head
  return `${path.slice(0, head)}…${path.slice(-tail)}`
}

export function SessionStart({
  onStart,
  onRecover,
  onRemove,
  listKey,
  loading,
  error,
}: Props) {
  const [path, setPath] = useState('')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [listLoading, setListLoading] = useState(true)
  const [listError, setListError] = useState<string | null>(null)
  const [removingId, setRemovingId] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    setListLoading(true)
    setListError(null)
    listSessions()
      .then(({ items }) => {
        if (!cancelled) setSessions(items)
      })
      .catch((e) => {
        if (!cancelled) {
          setListError(e instanceof Error ? e.message : String(e))
          setSessions([])
        }
      })
      .finally(() => {
        if (!cancelled) setListLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [listKey])

  function submit(e: FormEvent) {
    e.preventDefault()
    if (path.trim()) onStart(path.trim())
  }

  async function handleRemove(id: string) {
    setRemovingId(id)
    try {
      await onRemove(id)
    } finally {
      setRemovingId(null)
    }
  }

  const lastId = localStorage.getItem('picflow_session_id')

  return (
    <div className="session-start">
      <h1>PicFlow</h1>
      <p>Interactive image labeling with histogram features, clustering view, and ML suggestions.</p>

      <section className="session-list-section" aria-labelledby="saved-sessions-heading">
        <h2 id="saved-sessions-heading">Saved sessions</h2>
        {listLoading && <p className="muted">Loading sessions…</p>}
        {listError && <p className="error">{listError}</p>}
        {!listLoading && !listError && sessions.length === 0 && (
          <p className="muted">No saved sessions yet.</p>
        )}
        {!listLoading && sessions.length > 0 && (
          <ul className="session-list">
            {sessions.map((s) => (
              <li key={s.id} className="session-list-row">
                <div className="session-list-info">
                  <span className="session-list-path" title={s.image_root}>
                    {truncatePath(s.image_root)}
                  </span>
                  <span className="muted session-list-meta">
                    {s.labeled_count} / {s.total_images} labeled · {s.index_status}
                    {lastId === s.id ? ' · last used' : ''}
                  </span>
                </div>
                <div className="session-list-actions">
                  <button
                    type="button"
                    className="primary"
                    disabled={loading || removingId != null}
                    onClick={() => onRecover(s.id)}
                  >
                    Recover
                  </button>
                  <button
                    type="button"
                    className="danger"
                    disabled={loading || removingId === s.id}
                    onClick={() => handleRemove(s.id)}
                  >
                    {removingId === s.id ? 'Removing…' : 'Remove'}
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>

      <form onSubmit={submit}>
        <h2 className="new-session-heading">New session</h2>
        <label>
          Image folder on server
          <input
            type="text"
            value={path}
            onChange={(e) => setPath(e.target.value)}
            placeholder="/path/to/photos"
            disabled={loading}
          />
        </label>
        <button type="submit" disabled={loading || !path.trim()}>
          {loading ? 'Starting…' : 'Start session'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}
    </div>
  )
}
