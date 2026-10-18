import { useCallback, useEffect, useRef, useState } from 'react'
import {
  createSession,
  deleteSession,
  getImage,
  getImageIds,
  getProjection,
  getSession,
  getStats,
  getStatus,
  getSuggestion,
  listProviders,
  patchConfig,
  setLabel,
  trainSession,
  type ImageDetail,
  type Projection,
  type ProviderInfo,
  type SessionStatus,
  type SessionSummary,
  type Stats,
  type Suggestion,
} from './api'
import { LeftPanel } from './components/LeftPanel'
import { RightPanel } from './components/RightPanel'
import { SessionStart } from './components/SessionStart'
import { KEY_TO_LABEL, labelDisplayKey, nextUnusedLabelId } from './labels'
import './App.css'

function App() {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [session, setSession] = useState<SessionSummary | null>(null)
  const [status, setStatus] = useState<SessionStatus | null>(null)
  const [imageIds, setImageIds] = useState<number[]>([])
  const [cursor, setCursor] = useState(0)
  const [image, setImage] = useState<ImageDetail | null>(null)
  const [suggestion, setSuggestion] = useState<Suggestion | null>(null)
  const [projection, setProjection] = useState<Projection | null>(null)
  const [stats, setStats] = useState<Stats | null>(null)
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [unlabeledOnly, setUnlabeledOnly] = useState(false)
  const [configOpen, setConfigOpen] = useState(false)
  const [startLoading, setStartLoading] = useState(false)
  const [startError, setStartError] = useState<string | null>(null)
  const [sessionListKey, setSessionListKey] = useState(0)
  const pollRef = useRef<number | null>(null)

  const currentId = imageIds[cursor] ?? null

  const resetClientSession = useCallback(() => {
    localStorage.removeItem('picflow_session_id')
    setSessionId(null)
    setSession(null)
    setStatus(null)
    setImageIds([])
    setCursor(0)
    setImage(null)
    setSuggestion(null)
    setProjection(null)
    setStats(null)
    setUnlabeledOnly(false)
    setConfigOpen(false)
    if (pollRef.current) {
      window.clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  const bumpSessionList = useCallback(() => {
    setSessionListKey((k) => k + 1)
  }, [])

  const refreshIds = useCallback(async (sid: string, unlabeled: boolean) => {
    const { items } = await getImageIds(sid, unlabeled)
    setImageIds(items.map((i) => i.id))
    setCursor((c) => Math.min(c, Math.max(0, items.length - 1)))
  }, [])

  const refreshProjection = useCallback(async (sid: string, method?: 'pca' | 'umap') => {
    const proj = await getProjection(sid, method)
    setProjection(proj)
    if (proj.pending) {
      window.setTimeout(() => refreshProjection(sid, method), 800)
    }
  }, [])

  const refreshStats = useCallback(async (sid: string) => {
    setStats(await getStats(sid))
  }, [])

  const loadCurrentImage = useCallback(
    async (sid: string, id: number) => {
      const img = await getImage(sid, id)
      setImage(img)
      const sug = await getSuggestion(sid, id)
      setSuggestion(sug)
    },
    [],
  )

  useEffect(() => {
    listProviders().then(setProviders).catch(console.error)
  }, [])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false

    async function bootstrap() {
      try {
        const sess = await getSession(sessionId!)
        if (cancelled) return
        setSession(sess)
        localStorage.setItem('picflow_session_id', sessionId!)
        await refreshIds(sessionId!, unlabeledOnly)
        await refreshStats(sessionId!)
        await refreshProjection(sessionId!)
      } catch (e) {
        if (cancelled) return
        resetClientSession()
        setStartError(
          e instanceof Error ? e.message : 'Session not found or unavailable.',
        )
        bumpSessionList()
      }
    }

    bootstrap().catch(console.error)

    pollRef.current = window.setInterval(async () => {
      try {
        const st = await getStatus(sessionId!)
        setStatus(st)
        const sess = await getSession(sessionId!)
        setSession(sess)
        if (st.index_status === 'done') {
          await refreshIds(sessionId!, unlabeledOnly)
          await refreshStats(sessionId!)
          if (st.training_status === 'done' || st.projection_status === 'done') {
            await refreshProjection(sessionId!)
          }
        }
      } catch {
        /* ignore poll errors */
      }
    }, 1500)

    return () => {
      cancelled = true
      if (pollRef.current) window.clearInterval(pollRef.current)
    }
  }, [
    sessionId,
    unlabeledOnly,
    refreshIds,
    refreshProjection,
    refreshStats,
    resetClientSession,
    bumpSessionList,
  ])

  useEffect(() => {
    if (!sessionId || currentId == null) return
    loadCurrentImage(sessionId, currentId).catch(console.error)
  }, [sessionId, currentId, loadCurrentImage])

  function handleRecover(id: string) {
    setStartError(null)
    setSessionId(id)
  }

  function handleEndSession() {
    resetClientSession()
    bumpSessionList()
  }

  async function handleRemoveSession(id: string) {
    const msg =
      'Permanently delete this session and all labels? This cannot be undone.'
    if (!window.confirm(msg)) return
    setStartError(null)
    try {
      await deleteSession(id)
      if (sessionId === id) {
        resetClientSession()
      }
      bumpSessionList()
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e))
    }
  }

  async function handleStart(imageRoot: string) {
    setStartLoading(true)
    setStartError(null)
    try {
      const sess = await createSession(imageRoot)
      setSessionId(sess.id)
      setSession(sess)
      localStorage.setItem('picflow_session_id', sess.id)
      setCursor(0)
      bumpSessionList()
    } catch (e) {
      setStartError(e instanceof Error ? e.message : String(e))
    } finally {
      setStartLoading(false)
    }
  }

  async function applyLabel(label: number | null, source: 'user' | 'accepted_suggestion' = 'user') {
    if (!sessionId || currentId == null) return
    const updated = await setLabel(sessionId, currentId, label, source)
    setImage(updated)
    setSuggestion(await getSuggestion(sessionId, currentId))
    const sess = await getSession(sessionId)
    setSession(sess)
    await refreshStats(sessionId)
    await refreshProjection(sessionId)
    if (unlabeledOnly && label != null) {
      await refreshIds(sessionId, true)
      setCursor((c) => Math.min(c, Math.max(0, imageIds.length - 2)))
    }
  }

  function go(delta: number) {
    setCursor((c) => {
      const next = c + delta
      if (next < 0) return 0
      if (next >= imageIds.length) return Math.max(0, imageIds.length - 1)
      return next
    })
  }

  useEffect(() => {
    function onKeyDown(ev: KeyboardEvent) {
      if (ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement) return
      if (!sessionId) return

      if (ev.key === 'ArrowLeft' || ev.key === 'j') {
        ev.preventDefault()
        go(-1)
      } else if (ev.key === 'ArrowRight' || ev.key === 'k') {
        ev.preventDefault()
        go(1)
      } else if (ev.key === 'Backspace') {
        ev.preventDefault()
        applyLabel(null).catch(console.error)
      } else if (ev.key === 'Tab' || ev.key === 'Enter') {
        if (suggestion?.ready && suggestion.label != null && image?.label == null) {
          ev.preventDefault()
          applyLabel(suggestion.label, 'accepted_suggestion').catch(console.error)
        }
      } else if (KEY_TO_LABEL[ev.code] != null) {
        ev.preventDefault()
        applyLabel(KEY_TO_LABEL[ev.code]).catch(console.error)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [sessionId, suggestion, image, imageIds.length])

  async function handleSaveConfig(patch: Record<string, unknown>) {
    if (!sessionId) return
    const { config } = await patchConfig(sessionId, patch)
    setSession((s) => (s ? { ...s, config } : s))
    await refreshStats(sessionId)
  }

  const activeLabels = (() => {
    const raw = session?.config.active_labels
    if (Array.isArray(raw) && raw.length > 0) {
      return raw.map((v) => Number(v)).filter((n) => !Number.isNaN(n))
    }
    return [0, 1]
  })()

  const labelNames = (() => {
    const raw = session?.config.label_names
    if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
      return Object.fromEntries(
        Object.entries(raw as Record<string, unknown>).map(([k, v]) => [k, String(v)]),
      )
    }
    return {}
  })()

  const labelSlots =
    typeof session?.config.label_slots === 'number' ? session.config.label_slots : 10

  async function handleAddLabel() {
    if (!sessionId || !session) return
    const sorted = [...activeLabels].sort((a, b) => a - b)
    const nextId = nextUnusedLabelId(sorted, labelSlots)
    if (nextId == null) return
    const { config } = await patchConfig(sessionId, { active_labels: [...sorted, nextId] })
    setSession({ ...session, config })
  }

  async function handleRenameLabel(labelId: number, name: string) {
    if (!sessionId || !session) return
    const defaultName = `Class ${labelDisplayKey(labelId)}`
    const trimmed = name.trim()
    const nextNames = { ...labelNames }
    if (!trimmed || trimmed === defaultName) {
      delete nextNames[String(labelId)]
    } else {
      nextNames[String(labelId)] = trimmed
    }
    const { config } = await patchConfig(sessionId, { label_names: nextNames })
    setSession({ ...session, config })
  }

  if (!sessionId || !session) {
    return (
      <SessionStart
        onStart={handleStart}
        onRecover={handleRecover}
        onRemove={handleRemoveSession}
        listKey={sessionListKey}
        loading={startLoading}
        error={startError}
      />
    )
  }

  return (
    <div className="app-shell">
      <LeftPanel
        sessionId={sessionId}
        image={image}
        position={cursor}
        total={imageIds.length}
        suggestion={suggestion}
        status={status}
        unlabeledOnly={unlabeledOnly}
        onEndSession={handleEndSession}
        onToggleUnlabeled={() => {
          setUnlabeledOnly((v) => !v)
          if (sessionId) refreshIds(sessionId, !unlabeledOnly).catch(console.error)
        }}
        onPrev={() => go(-1)}
        onNext={() => go(1)}
        onAcceptSuggestion={() => {
          if (suggestion?.label != null) {
            applyLabel(suggestion.label, 'accepted_suggestion').catch(console.error)
          }
        }}
        activeLabels={activeLabels}
        labelNames={labelNames}
        labelSlots={labelSlots}
        onAddLabel={() => handleAddLabel().catch(console.error)}
        onRenameLabel={(id, name) => handleRenameLabel(id, name).catch(console.error)}
      />
      <RightPanel
        projection={projection}
        stats={stats}
        currentId={currentId}
        session={session}
        providers={providers}
        configOpen={configOpen}
        onOpenConfig={() => setConfigOpen(true)}
        onCloseConfig={() => setConfigOpen(false)}
        onSaveConfig={handleSaveConfig}
        onRecomputeUmap={() => sessionId && refreshProjection(sessionId, 'umap')}
        onTrain={() => sessionId && trainSession(sessionId).then(() => refreshStats(sessionId!))}
        onSelectImage={(id) => {
          const idx = imageIds.indexOf(id)
          if (idx >= 0) setCursor(idx)
        }}
        activeLabels={activeLabels}
        labelNames={labelNames}
      />
    </div>
  )
}

export default App
