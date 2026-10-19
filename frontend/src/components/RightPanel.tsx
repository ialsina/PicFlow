import type { LabelImportResult, Projection, Stats } from '../api'
import { ConfigDrawer } from './ConfigDrawer'
import { ScatterPlot } from './ScatterPlot'
import { StatsPanel } from './StatsPanel'
import type { ProviderInfo, SessionSummary } from '../api'

type Props = {
  projection: Projection | null
  stats: Stats | null
  currentId: number | null
  session: SessionSummary
  providers: ProviderInfo[]
  configOpen: boolean
  onOpenConfig: () => void
  onCloseConfig: () => void
  onSaveConfig: (patch: Record<string, unknown>) => Promise<void>
  onRecomputeUmap: () => void
  onTrain: () => void
  onLabelsImported: (result: LabelImportResult) => Promise<void>
  onSelectImage: (id: number) => void
  activeLabels: number[]
  labelNames: Record<string, string>
}

export function RightPanel({
  projection,
  stats,
  currentId,
  session,
  providers,
  configOpen,
  onOpenConfig,
  onCloseConfig,
  onSaveConfig,
  onRecomputeUmap,
  onTrain,
  onLabelsImported,
  onSelectImage,
  activeLabels,
  labelNames,
}: Props) {
  return (
    <section className="panel right-panel">
      <header className="panel-header">
        <h2>Clusters &amp; stats</h2>
        <button type="button" onClick={onOpenConfig}>
          Settings
        </button>
      </header>
      <div className="scatter-wrap">
        {projection && !projection.pending ? (
          <ScatterPlot
            points={projection.points}
            currentId={currentId}
            method={projection.method}
            explainedVariance={projection.explained_variance}
            onSelect={onSelectImage}
          />
        ) : (
          <p className="muted">Computing layout…</p>
        )}
      </div>
      <StatsPanel stats={stats} activeLabels={activeLabels} labelNames={labelNames} />
      <ConfigDrawer
        session={session}
        providers={providers}
        open={configOpen}
        onClose={onCloseConfig}
        onSave={onSaveConfig}
        onRecomputeUmap={onRecomputeUmap}
        onTrain={onTrain}
        onLabelsImported={onLabelsImported}
      />
    </section>
  )
}
