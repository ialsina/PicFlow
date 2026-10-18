import { useEffect, useRef, useState } from 'react'
import type { ImageDetail, SessionStatus, Suggestion } from '../api'
import { LABEL_COLORS, labelDisplayKey, labelName, nextUnusedLabelId } from '../labels'

type Props = {
  sessionId: string
  image: ImageDetail | null
  position: number
  total: number
  suggestion: Suggestion | null
  status: SessionStatus | null
  unlabeledOnly: boolean
  activeLabels: number[]
  labelNames: Record<string, string>
  labelSlots: number
  onToggleUnlabeled: () => void
  onEndSession: () => void
  onPrev: () => void
  onNext: () => void
  onAcceptSuggestion: () => void
  onAddLabel: () => void
  onRenameLabel: (labelId: number, name: string) => void
}

function LabelNameCell({
  labelId,
  labelNames,
  onRename,
}: {
  labelId: number
  labelNames: Record<string, string>
  onRename: (name: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  const display = labelName(labelId, labelNames)

  useEffect(() => {
    if (editing) inputRef.current?.focus()
  }, [editing])

  function startEdit() {
    setDraft(display)
    setEditing(true)
  }

  function commit() {
    setEditing(false)
    const trimmed = draft.trim()
    onRename(trimmed || labelName(labelId))
  }

  function cancel() {
    setEditing(false)
    setDraft(display)
  }

  if (editing) {
    return (
      <input
        ref={inputRef}
        className="legend-name-input"
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            commit()
          } else if (e.key === 'Escape') {
            e.preventDefault()
            cancel()
          }
        }}
      />
    )
  }

  return (
    <button type="button" className="legend-name-btn" onClick={startEdit} title="Click to rename">
      {display}
    </button>
  )
}

export function LeftPanel({
  sessionId,
  image,
  position,
  total,
  suggestion,
  status,
  unlabeledOnly,
  activeLabels,
  labelNames,
  labelSlots,
  onToggleUnlabeled,
  onEndSession,
  onPrev,
  onNext,
  onAcceptSuggestion,
  onAddLabel,
  onRenameLabel,
}: Props) {
  const imgUrl = image
    ? `/api/sessions/${sessionId}/images/${image.id}/file`
    : null

  const sortedLabels = [...activeLabels].sort((a, b) => a - b)
  const canAdd = nextUnusedLabelId(sortedLabels, labelSlots) != null

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
                {labelName(image.label, labelNames)}
              </span>
            )}
          </span>
        </div>
        <div className="nav-buttons">
          <button type="button" className="end-session" onClick={onEndSession}>
            End session
          </button>
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
              {labelName(suggestion.label, labelNames)}
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
        <div className="legend-grid">
          {sortedLabels.map((id) => (
              <div key={id} className="legend-cell">
                <kbd>{labelDisplayKey(id)}</kbd>
                <span className="swatch" style={{ background: LABEL_COLORS[id] }} />
                <LabelNameCell
                  labelId={id}
                  labelNames={labelNames}
                  onRename={(name) => onRenameLabel(id, name)}
                />
              </div>
          ))}
          {canAdd && (
            <div className="legend-cell legend-add-cell">
              <button type="button" className="legend-add-btn" onClick={onAddLabel}>
                + Add
              </button>
            </div>
          )}
        </div>
        <p className="muted keys-hint">← → or j/k navigate · Backspace clears label</p>
      </div>
    </section>
  )
}
