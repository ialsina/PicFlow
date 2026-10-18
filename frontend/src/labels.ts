/** Keyboard keys shown in UI map to internal label ids 0–9 */
export const LABEL_COLORS = [
  '#1f77b4',
  '#ff7f0e',
  '#2ca02c',
  '#d62728',
  '#9467bd',
  '#8c564b',
  '#e377c2',
  '#7f7f7f',
  '#bcbd22',
  '#17becf',
]

export const KEY_TO_LABEL: Record<string, number> = {
  Digit1: 0,
  Digit2: 1,
  Digit3: 2,
  Digit4: 3,
  Digit5: 4,
  Digit6: 5,
  Digit7: 6,
  Digit8: 7,
  Digit9: 8,
  Digit0: 9,
}

export function labelDisplayKey(labelId: number): string {
  return labelId === 9 ? '0' : String(labelId + 1)
}

export function labelName(labelId: number, labelNames?: Record<string, string>): string {
  const custom = labelNames?.[String(labelId)]?.trim()
  if (custom) return custom
  return `Class ${labelDisplayKey(labelId)}`
}

export function nextUnusedLabelId(active: number[], slots = 10): number | null {
  const used = new Set(active)
  for (let id = 0; id < slots; id++) {
    if (!used.has(id)) return id
  }
  return null
}
