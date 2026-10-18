import Plot from 'react-plotly.js'
import type { Data } from 'plotly.js'
import type { ProjectionPoint } from '../api'
import { LABEL_COLORS } from '../labels'

type Props = {
  points: ProjectionPoint[]
  currentId: number | null
  method: string
  explainedVariance?: number[] | null
  onSelect: (id: number) => void
}

export function ScatterPlot({
  points,
  currentId,
  method,
  explainedVariance,
  onSelect,
}: Props) {
  if (!points.length) {
    return <p className="muted">Projection loading…</p>
  }

  const labeled = points.filter((p) => p.label != null)
  const unlabeled = points.filter((p) => p.label == null)

  const title =
    method === 'pca' && explainedVariance?.length
      ? `PCA (${(explainedVariance[0] * 100).toFixed(0)}% / ${(explainedVariance[1] * 100).toFixed(0)}% variance)`
      : method.toUpperCase()

  const traces: Data[] = []

  for (let lab = 0; lab < 10; lab++) {
    const pts = labeled.filter((p) => p.label === lab)
    if (!pts.length) continue
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      name: `Class ${lab === 9 ? '0' : lab + 1}`,
      x: pts.map((p) => p.x),
      y: pts.map((p) => p.y),
      marker: { color: LABEL_COLORS[lab], size: 7, opacity: 0.85 },
      customdata: pts.map((p) => String(p.id)),
      hovertemplate: 'idx %{text}<extra></extra>',
      text: pts.map((p) => String(p.idx)),
    })
  }

  if (unlabeled.length) {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      name: 'Unlabeled',
      x: unlabeled.map((p) => p.x),
      y: unlabeled.map((p) => p.y),
      marker: { color: '#ccc', size: 5, opacity: 0.5 },
      customdata: unlabeled.map((p) => String(p.id)),
    })
  }

  const current = points.find((p) => p.id === currentId)
  if (current) {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      name: 'Current',
      x: [current.x],
      y: [current.y],
      marker: {
        color: '#000',
        size: 14,
        symbol: 'circle-open',
        line: { width: 2 },
      },
      showlegend: false,
    })
  }

  return (
    <Plot
      data={traces}
      layout={{
        title: { text: title, font: { size: 14 } },
        autosize: true,
        margin: { l: 40, r: 20, t: 40, b: 40 },
        paper_bgcolor: 'transparent',
        plot_bgcolor: '#fafafa',
        legend: { orientation: 'h', y: -0.15 },
        xaxis: { zeroline: false },
        yaxis: { zeroline: false },
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100%' }}
      useResizeHandler
      onClick={(ev) => {
        const raw = ev.points?.[0]?.customdata
        const id = typeof raw === 'number' ? raw : Number(raw)
        if (!Number.isNaN(id)) onSelect(id)
      }}
    />
  )
}
