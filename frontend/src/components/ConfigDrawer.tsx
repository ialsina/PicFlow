import { useEffect, useState, type FormEvent } from 'react'
import type { ProviderInfo, SessionSummary } from '../api'

type Props = {
  session: SessionSummary
  providers: ProviderInfo[]
  open: boolean
  onClose: () => void
  onSave: (patch: Record<string, unknown>) => Promise<void>
  onRecomputeUmap: () => void
  onTrain: () => void
}

export function ConfigDrawer({
  session,
  providers,
  open,
  onClose,
  onSave,
  onRecomputeUmap,
  onTrain,
}: Props) {
  const cfg = session.config
  const [minTotal, setMinTotal] = useState(String(cfg.min_labeled_total ?? 30))
  const [minPerClass, setMinPerClass] = useState(String(cfg.min_labeled_per_class ?? 3))
  const [provider, setProvider] = useState(String(cfg.suggestion_provider ?? 'sklearn_mlp'))
  const [mlpIter, setMlpIter] = useState(
    String((cfg.sklearn_mlp as { max_iter?: number })?.max_iter ?? 200),
  )
  const [knnK, setKnnK] = useState(String((cfg.knn as { n_neighbors?: number })?.n_neighbors ?? 5))
  const [ptEpochs, setPtEpochs] = useState(
    String((cfg.pytorch_mlp as { epochs?: number })?.epochs ?? 50),
  )

  useEffect(() => {
    setMinTotal(String(session.config.min_labeled_total ?? 30))
    setMinPerClass(String(session.config.min_labeled_per_class ?? 3))
    setProvider(String(session.config.suggestion_provider ?? 'sklearn_mlp'))
  }, [session])

  if (!open) return null

  async function submit(e: FormEvent) {
    e.preventDefault()
    await onSave({
      min_labeled_total: Number(minTotal),
      min_labeled_per_class: Number(minPerClass),
      suggestion_provider: provider,
      sklearn_mlp: { max_iter: Number(mlpIter) },
      knn: { n_neighbors: Number(knnK) },
      pytorch_mlp: { epochs: Number(ptEpochs) },
    })
    onClose()
  }

  return (
    <div className="drawer-backdrop" onClick={onClose} role="presentation">
      <aside className="drawer" onClick={(e) => e.stopPropagation()}>
        <header>
          <h2>Settings</h2>
          <button type="button" onClick={onClose}>
            ×
          </button>
        </header>
        <form onSubmit={submit}>
          <label>
            Min labeled (total)
            <input type="number" min={1} value={minTotal} onChange={(e) => setMinTotal(e.target.value)} />
          </label>
          <label>
            Min per class
            <input
              type="number"
              min={1}
              value={minPerClass}
              onChange={(e) => setMinPerClass(e.target.value)}
            />
          </label>
          <label>
            Suggestion provider
            <select value={provider} onChange={(e) => setProvider(e.target.value)}>
              {providers.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            sklearn MLP max_iter (epochs)
            <input type="number" min={1} value={mlpIter} onChange={(e) => setMlpIter(e.target.value)} />
          </label>
          <label>
            kNN neighbors
            <input type="number" min={1} value={knnK} onChange={(e) => setKnnK(e.target.value)} />
          </label>
          <label>
            PyTorch epochs
            <input type="number" min={1} value={ptEpochs} onChange={(e) => setPtEpochs(e.target.value)} />
          </label>
          <div className="drawer-actions">
            <button type="submit">Save</button>
            <button type="button" onClick={onRecomputeUmap}>
              Recompute UMAP
            </button>
            <button type="button" onClick={onTrain}>
              Train now
            </button>
          </div>
        </form>
      </aside>
    </div>
  )
}
