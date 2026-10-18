import type { ImageDetail, SessionStatus, Suggestion } from '../api'
import { LABEL_COLORS, labelDisplayKey, labelName } from '../labels'

type Props = {
  sessionId: string
  image: ImageDetail | null
  position: number
  total: number
  suggestion: Suggestion | null
  status: SessionStatus | null
  unlabeledOnly: boolean
  onToggleUnlabeled: () => void
  onPrev: () => void
  onNext: () => void
  onAcceptSuggestion: () => void
}

export function LeftPanel({
  sessionId,
  image,
  position,
  total,
  suggestion,
  status,
  unlabeledOnly,
  onToggleUnlabeled,
  onPrev,
  onNext,
  onAcceptSuggestion,
}: Props) {
  const imgUrl = image
    ? `/api/sessions/${sessionId}/images/${image.id}/file`
    : null

  return (
    <section className="panel left-panel">
      <header className="panel-header">
        <div>
          <h2>{image?.filename ?? '—'}</h2>
          <span className="muted">
            {total ? position + 1 : 0} / {total}
            {image?.label != null && (
              <span
                className="current-label"
                style={{ background: LABEL_COLORS[image.label] }}
              >
                {labelName(image.label)}
              </span>
            )}
          </span>
        </div>
        <div className="nav-buttons">
          <label className="checkbox">
            <input type="checkbox" checked={unlabeledOnly} onChange={onToggleUnlabeled} />
            Unlabeled only
          </label>
          <button type="button" onClick={onPrev}>
            ← Prev
          </button>
          <button type="button" onClick={onNext}>
            Next →
          </button>
        </div>
      </header>

      {status && status.index_status !== 'done' && (
        <div className="progress-bar">
          <div className="fill" style={{ width: `${status.index_progress * 100}%` }} />
          <span>{status.index_message}</span>
        </div>
      )}

      <div className="image-stage">
        {imgUrl ? (
          <img src={imgUrl} alt={image?.filename ?? ''} />
        ) : (
          <p className="muted">No image selected</p>
        )}
      </div>

      {suggestion?.ready && suggestion.label != null && (
        <div className="suggestion-strip">
          <span>
            Suggestion:{' '}
            <strong style={{ color: LABEL_COLORS[suggestion.label] }}>
              {labelName(suggestion.label)}
            </strong>
            {suggestion.confidence != null && (
              <span className="muted"> ({(suggestion.confidence * 100).toFixed(0)}%)</span>
            )}
            <span className="muted"> via {suggestion.provider}</span>
          </span>
          <button type="button" onClick={onAcceptSuggestion}>
            Accept (Tab / Enter)
          </button>
        </div>
      )}

      {!suggestion?.ready && suggestion?.reason && image?.label == null && (
        <p className="muted suggestion-hint">{suggestion.reason}</p>
      )}

      <div className="legend">
        <h3>Labels</h3>
        <ul>
          {LABEL_COLORS.map((color, id) => (
            <li key={id}>
              <kbd>{labelDisplayKey(id)}</kbd>
              <span className="swatch" style={{ background: color }} />
              {labelName(id)}
            </li>
          ))}
        </ul>
        <p className="muted keys-hint">← → or j/k navigate · Backspace clears label</p>
      </div>
    </section>
  )
}
