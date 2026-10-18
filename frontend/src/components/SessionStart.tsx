import { useState, type FormEvent } from 'react'

type Props = {
  onStart: (imageRoot: string) => void
  loading: boolean
  error: string | null
}

export function SessionStart({ onStart, loading, error }: Props) {
  const [path, setPath] = useState('')

  function submit(e: FormEvent) {
    e.preventDefault()
    if (path.trim()) onStart(path.trim())
  }

  return (
    <div className="session-start">
      <h1>PicFlow</h1>
      <p>Interactive image labeling with histogram features, clustering view, and ML suggestions.</p>
      <form onSubmit={submit}>
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
