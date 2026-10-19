import { useEffect, useRef, useState, type FormEvent } from 'react'
import {
  exportLabels,
  importLabels,
  type LabelImportResult,
  type ProviderInfo,
  type SessionSummary,
} from '../api'

type Props = {
  session: SessionSummary
  providers: ProviderInfo[]
  open: boolean
  onClose: () => void
  onSave: (patch: Record<string, unknown>) => Promise<void>
  onRecomputeUmap: () => void
  onTrain: () => void
  onLabelsImported: (result: LabelImportResult) => Promise<void>
}

export function ConfigDrawer({
  session,
  providers,
  open,
  onClose,
  onSave,
  onRecomputeUmap,
  onTrain,
  onLabelsImported,
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
  const [labelIoBusy, setLabelIoBusy] = useState(false)
  const [labelIoMessage, setLabelIoMessage] = useState<string | null>(null)
  const archiveRef = useRef<HTMLInputElement>(null)
  const labelsRef = useRef<HTMLInputElement>(null)
  const namesRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setMinTotal(String(session.config.min_labeled_total ?? 30))
    setMinPerClass(String(session.config.min_labeled_per_class ?? 3))
    setProvider(String(session.config.suggestion_provider ?? 'sklearn_mlp'))
  }, [session])

  useEffect(() => {
    if (open) {
      setLabelIoMessage(null)
    }
  }, [open, session.id])

  if (!open) return null

  const canExport = session.labeled_count > 0 && !labelIoBusy

  function formatImportResult(result: LabelImportResult): string {
    const parts = [`Applied ${result.applied} label(s).`]
    if (result.skipped_unknown) parts.push(`${result.skipped_unknown} unknown image(s).`)
    if (result.skipped_ambiguous) parts.push(`${result.skipped_ambiguous} ambiguous name(s).`)
    if (result.skipped_invalid) parts.push(`${result.skipped_invalid} invalid row(s).`)
    if (result.config_updated) parts.push('Label names updated.')
    if (result.errors.length) {
      parts.push(result.errors.slice(0, 3).join(' '))
    }
    return parts.join(' ')
  }

  async function handleExportLabels() {
    setLabelIoBusy(true)
    setLabelIoMessage(null)
    try {
      await exportLabels(session.id)
      setLabelIoMessage('Export downloaded.')
    } catch (e) {
      setLabelIoMessage(e instanceof Error ? e.message : 'Export failed')
    } finally {
      setLabelIoBusy(false)
    }
  }

  async function handleImportLabels() {
    const archive = archiveRef.current?.files?.[0]
    const labelsFile = labelsRef.current?.files?.[0]
    const namesFile = namesRef.current?.files?.[0]
    if (!archive && !labelsFile) {
      setLabelIoMessage('Choose a ZIP archive or labels.csv file.')
      return
    }
    const form = new FormData()
    if (archive) {
      form.append('archive', archive)
    } else if (labelsFile) {
      form.append('labels', labelsFile)
      if (namesFile) {
        form.append('label_names', namesFile)
      }
    }
    setLabelIoBusy(true)
    setLabelIoMessage(null)
    try {
      const result = await importLabels(session.id, form)
      await onLabelsImported(result)
      setLabelIoMessage(formatImportResult(result))
      if (archiveRef.current) archiveRef.current.value = ''
      if (labelsRef.current) labelsRef.current.value = ''
      if (namesRef.current) namesRef.current.value = ''
    } catch (e) {
      setLabelIoMessage(e instanceof Error ? e.message : 'Import failed')
    } finally {
      setLabelIoBusy(false)
    }
  }

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
          <fieldset className="drawer-fieldset">
            <legend>Labels (CSV)</legend>
            <p className="muted drawer-hint">
              Export labeled images as a ZIP (<code>labels.csv</code> + <code>label_names.csv</code>).
              While indexing is still running, only images already indexed and labeled are included.
              Import into this session from a ZIP or separate CSV files.
            </p>
            <div className="drawer-actions">
              <button type="button" disabled={!canExport} onClick={() => handleExportLabels()}>
                Export labels
              </button>
            </div>
            <label>
              ZIP archive
              <input ref={archiveRef} type="file" accept=".zip,application/zip" disabled={labelIoBusy} />
            </label>
            <label>
              labels.csv
              <input ref={labelsRef} type="file" accept=".csv,text/csv" disabled={labelIoBusy} />
            </label>
            <label>
              label_names.csv (optional)
              <input ref={namesRef} type="file" accept=".csv,text/csv" disabled={labelIoBusy} />
            </label>
            <div className="drawer-actions">
              <button type="button" disabled={labelIoBusy} onClick={() => handleImportLabels()}>
                Import labels
              </button>
            </div>
            {labelIoMessage ? <p className="drawer-hint">{labelIoMessage}</p> : null}
          </fieldset>
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
