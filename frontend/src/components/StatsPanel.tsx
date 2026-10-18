import type { Stats } from '../api'
import { labelName } from '../labels'

type Props = {
  stats: Stats | null
  activeLabels: number[]
  labelNames: Record<string, string>
}

export function StatsPanel({ stats, activeLabels, labelNames }: Props) {
  if (!stats) return <p className="muted">Loading stats…</p>

  const maxCount = Math.max(1, ...Object.values(stats.per_label))

  return (
    <div className="stats-panel">
      <div className="stat-grid">
        <div>
          <span className="stat-value">{stats.labeled}</span>
          <span className="stat-label">Labeled</span>
        </div>
        <div>
          <span className="stat-value">{stats.unlabeled}</span>
          <span className="stat-label">Unlabeled</span>
        </div>
        <div>
          <span className="stat-value">{stats.total}</span>
          <span className="stat-label">Total</span>
        </div>
      </div>

      <h3>Per class</h3>
      <ul className="bar-list">
        {[...activeLabels]
          .sort((a, b) => a - b)
          .map((lab) => [String(lab), stats.per_label[String(lab)] ?? 0] as const)
          .map(([lab, count]) => (
          <li key={lab}>
            <span className="bar-label">{labelName(Number(lab), labelNames)}</span>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${(count / maxCount) * 100}%` }} />
            </div>
            <span className="bar-count">{count}</span>
          </li>
        ))}
      </ul>

      <h3>Model</h3>
      <ul className="model-stats">
        <li>
          Status: <strong>{stats.model_ready ? 'Ready' : stats.training_status}</strong>
        </li>
        <li>Provider: {stats.provider ?? '—'}</li>
        <li>
          Holdout accuracy:{' '}
          {stats.holdout_accuracy != null ? `${(stats.holdout_accuracy * 100).toFixed(1)}%` : '—'}
        </li>
        <li>
          Last train:{' '}
          {stats.last_train_ms != null ? `${stats.last_train_ms.toFixed(0)} ms` : '—'}
        </li>
        <li>Projection: {stats.projection_method ?? '—'}</li>
      </ul>
    </div>
  )
}
