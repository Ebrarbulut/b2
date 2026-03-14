import { useState, useEffect, useRef } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''
const isLocal =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')

function buildUrl(path: string) {
  if (API_BASE) return `${API_BASE}${path}`
  return path
}

const modelLabels: Record<string, string> = {
  standard_ae: 'Standard Autoencoder',
  isolation_forest: 'Isolation Forest',
  one_class_svm: 'One-Class SVM',
  lstm_ae: 'LSTM Autoencoder',
  ensemble: 'Ensemble',
  demo: 'Demo',
}

const fontFamily = "'Plus Jakarta Sans', system-ui, -apple-system, sans-serif"

const makeCardStyle = (theme: 'light' | 'dark') =>
  ({
    marginTop: 28,
    padding: '28px 26px',
    borderRadius: 22,
    fontFamily,
    background:
      theme === 'light'
        ? 'linear-gradient(160deg, #ffffff 0%, #fafaff 35%, #f5f3ff 100%)'
        : 'linear-gradient(160deg, rgba(18,18,22,0.98) 0%, rgba(24,24,32,0.95) 100%)',
    border:
      theme === 'light'
        ? '1px solid rgba(99,102,241,0.18)'
        : '1px solid rgba(129,140,248,0.15)',
    boxShadow:
      theme === 'light'
        ? '0 1px 2px rgba(0,0,0,0.04), 0 12px 40px -8px rgba(99,102,241,0.12), 0 0 0 1px rgba(255,255,255,0.8) inset'
        : '0 1px 0 rgba(255,255,255,0.04), 0 12px 40px -8px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.03) inset',
  } as const)

const makeTitleStyle = (theme: 'light' | 'dark') =>
  ({
    margin: '0 0 8px 0',
    fontSize: 20,
    fontWeight: 800,
    letterSpacing: '-0.02em',
    fontFamily,
    background:
      theme === 'light'
        ? 'linear-gradient(105deg, #6366f1 0%, #8b5cf6 50%, #c026d3 100%)'
        : 'linear-gradient(105deg, #a5b4fc 0%, #c4b5fd 50%, #e879f9 100%)',
    WebkitBackgroundClip: 'text' as const,
    color: 'transparent',
  })

const makeDescStyle = (theme: 'light' | 'dark') =>
  ({
    color: theme === 'light' ? '#475569' : '#94a3b8',
    fontSize: 15,
    marginBottom: 16,
    lineHeight: 1.6,
    fontWeight: 500,
    fontFamily,
  } as const)

const themeColors = (theme: 'light' | 'dark') => ({
  text: theme === 'light' ? '#0f172a' : '#f1f5f9',
  textMuted: theme === 'light' ? '#64748b' : '#94a3b8',
  cardBg: theme === 'light' ? 'rgba(248,250,252,0.9)' : 'rgba(30,30,40,0.85)',
  inputBg: theme === 'light' ? '#f8fafc' : 'rgba(30,30,40,0.9)',
  inputBorder: theme === 'light' ? 'rgba(148,163,184,0.35)' : 'rgba(148,163,184,0.25)',
  boxBg: theme === 'light' ? '#f1f5f9' : 'rgba(30,30,40,0.7)',
  preBg: theme === 'light' ? '#1e293b' : '#0f172a',
  preText: theme === 'light' ? '#e2e8f0' : '#cbd5e1',
  accent: theme === 'light' ? '#6366f1' : '#818cf8',
  success: '#059669',
  danger: '#dc2626',
  // Kod blokları için yumuşak stil (siyah/koyu yerler soft)
  codeSoftBg: theme === 'light' ? 'rgba(100,116,139,0.12)' : 'rgba(148,163,184,0.12)',
  codeSoftText: theme === 'light' ? '#475569' : '#e2e8f0',
})

export default function App() {
  const [health, setHealth] = useState<string>('')
  const [scores, setScores] = useState<number[] | null>(null)
  const [selectedModel, setSelectedModel] = useState<string>('auto')
  const [healthStatus, setHealthStatus] = useState<'up' | 'down' | null>(null)
  const [lastCheck, setLastCheck] = useState<string | null>(null)
  const [isChecking, setIsChecking] = useState(false)
  const [isScoring, setIsScoring] = useState(false)
  const [scoreNote, setScoreNote] = useState<string>('')
  const [usedModelName, setUsedModelName] = useState<string>('')
  const [apiModels, setApiModels] = useState<{ models: string[]; ensemble_available: boolean } | null>(null)
  const [comparison, setComparison] = useState<{ available: boolean; rows: Record<string, unknown>[] } | null>(null)
  const [installable, setInstallable] = useState(false)
  const [checkIntervalMinutes, setCheckIntervalMinutes] = useState(1)
  const [backgroundMode, setBackgroundMode] = useState<'off' | 'interval' | 'hours' | 'always'>('off')
  const [hourStart, setHourStart] = useState(9)
  const [hourEnd, setHourEnd] = useState(18)
  const [showHelp, setShowHelp] = useState(false)
  const [csvSummary, setCsvSummary] = useState<{
    model: string
    total_rows: number
    anomaly_count: number
    anomaly_ratio: number
    score_min: number
    score_max: number
    score_mean: number
  } | null>(null)
  const [csvError, setCsvError] = useState<string | null>(null)
  const [csvLoading, setCsvLoading] = useState(false)
  const csvInputRef = useRef<HTMLInputElement | null>(null)
  const [pcapSummary, setPcapSummary] = useState<{
    model: string
    total_rows: number
    anomaly_count: number
    anomaly_ratio: number
    score_min: number
    score_max: number
    score_mean: number
  } | null>(null)
  const [pcapError, setPcapError] = useState<string | null>(null)
  const [pcapLoading, setPcapLoading] = useState(false)
  const pcapInputRef = useRef<HTMLInputElement | null>(null)
  const [theme, setTheme] = useState<'light' | 'dark'>('light')
  const installPromptRef = useRef<{ prompt: () => void } | null>(null)

  useEffect(() => {
    const handler = (e: Event) => {
      e.preventDefault()
      installPromptRef.current = (e as { prompt: () => void })
      setInstallable(true)
    }
    window.addEventListener('beforeinstallprompt', handler)
    return () => window.removeEventListener('beforeinstallprompt', handler)
  }, [])

  const fetchModels = async () => {
    try {
      const r = await fetch(buildUrl('/api/models'))
      if (r.ok) setApiModels(await r.json())
      else setApiModels(null)
    } catch {
      setApiModels(null)
    }
  }

  const fetchComparison = async () => {
    try {
      const r = await fetch(buildUrl('/api/comparison'))
      if (r.ok) setComparison(await r.json())
      else setComparison(null)
    } catch {
      setComparison(null)
    }
  }

  const checkHealth = async () => {
    try {
      setIsChecking(true)
      const r = await fetch(buildUrl('/health'))
      const text = await r.text()
      if (!r.ok) {
        setHealth(`Hata ${r.status}: ${text.slice(0, 200)}`)
        setHealthStatus('down')
        setLastCheck(new Date().toLocaleTimeString())
        return
      }
      const j = JSON.parse(text)
      setHealth(JSON.stringify(j, null, 2))
      setHealthStatus('up')
      setLastCheck(new Date().toLocaleTimeString())
    } catch (e) {
      setHealth('Hata: ' + String(e))
      setHealthStatus('down')
      setLastCheck(new Date().toLocaleTimeString())
    } finally {
      setIsChecking(false)
    }
  }

  const testScore = async () => {
    try {
      setIsScoring(true)
      setScoreNote('')
      setUsedModelName('')
      const r = await fetch(buildUrl('/api/score'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: [[0, 100, 200, 5, 10, 300, 15, 0.5, 0.5, 0.1, 0.2, 2]],
          model: selectedModel === 'auto' ? null : selectedModel,
        }),
      })
      const text = await r.text()
      if (!r.ok) {
        setHealth(`Skor hatası ${r.status}: ${text.slice(0, 200)}`)
        setScores(null)
        return
      }
      const j = JSON.parse(text)
      setScores(j.scores || null)
      setUsedModelName(j.model || '')
      if (j.model === 'demo') {
        setScoreNote('Sunucuda model dosyası yok; örnek skorlar dönüyor. Lokalde compare_core_models.py çalıştırın.')
      } else {
        const s = Array.isArray(j.scores) && j.scores.length > 0 ? j.scores[0] : null
        if (s == null) setScoreNote('')
        else if (typeof s === 'number' && s < 0.3) setScoreNote('Düşük risk: Örnek trafik normal aralıkta.')
        else if (typeof s === 'number' && s < 0.7) setScoreNote('Orta risk: Örnek trafik şüpheli olabilir.')
        else setScoreNote('Yüksek risk: Örnek trafik anomali olarak işaretlendi.')
      }
    } catch (e) {
      setScores(null)
      setHealth('Skor hatası: ' + String(e))
    } finally {
      setIsScoring(false)
    }
  }

  const handleInstall = () => {
    if (installPromptRef.current) {
      installPromptRef.current.prompt()
      setInstallable(false)
    }
  }

  const analyzeCsv = async () => {
    const file = csvInputRef.current?.files?.[0]
    if (!file) {
      setCsvError('Lütfen bir CSV dosyası seçin.')
      return
    }
    setCsvError(null)
    setCsvSummary(null)
    setCsvLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      if (selectedModel !== 'auto') {
        form.append('model', selectedModel)
      }
      const r = await fetch(buildUrl('/api/analyze-csv'), {
        method: 'POST',
        body: form,
      })
      const text = await r.text()
      if (!r.ok) {
        setCsvError(text.slice(0, 300) || 'Analiz sırasında hata oluştu.')
        return
      }
      const j = JSON.parse(text)
      setCsvSummary(j)
    } catch (e) {
      setCsvError('İstek hatası: ' + String(e))
    } finally {
      setCsvLoading(false)
    }
  }

  const analyzePcap = async () => {
    const file = pcapInputRef.current?.files?.[0]
    if (!file) {
      setPcapError('Lütfen bir PCAP dosyası seçin.')
      return
    }
    setPcapError(null)
    setPcapSummary(null)
    setPcapLoading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      if (selectedModel !== 'auto') {
        form.append('model', selectedModel)
      }
      const r = await fetch(buildUrl('/api/analyze-pcap'), {
        method: 'POST',
        body: form,
      })
      const text = await r.text()
      if (!r.ok) {
        setPcapError(text.slice(0, 300) || 'Analiz sırasında hata oluştu.')
        return
      }
      const j = JSON.parse(text)
      setPcapSummary(j)
    } catch (e) {
      setPcapError('İstek hatası: ' + String(e))
    } finally {
      setPcapLoading(false)
    }
  }

  useEffect(() => {
    checkHealth()
    fetchModels()
    fetchComparison()
  }, [])

  useEffect(() => {
    if (backgroundMode === 'off') return
    if (backgroundMode === 'always') {
      const id = window.setInterval(checkHealth, 5 * 60 * 1000)
      return () => clearInterval(id)
    }
    if (backgroundMode === 'interval') {
      const ms = checkIntervalMinutes * 60 * 1000
      const id = window.setInterval(checkHealth, ms)
      return () => clearInterval(id)
    }
    if (backgroundMode === 'hours') {
      const id = window.setInterval(() => {
        const h = new Date().getHours()
        const inRange = hourStart <= hourEnd ? (h >= hourStart && h <= hourEnd) : (h >= hourStart || h <= hourEnd)
        if (inRange) checkHealth()
      }, 60 * 1000)
      return () => clearInterval(id)
    }
  }, [backgroundMode, checkIntervalMinutes, hourStart, hourEnd])

  const colors = themeColors(theme)
  const bgStyle =
    theme === 'light'
      ? {
          minHeight: '100vh',
          fontFamily,
          background:
            'linear-gradient(180deg, #f8fafc 0%, #f1f5f9 30%, #eef2ff 60%, #e0e7ff 100%)',
          color: '#0f172a',
          padding: '32px 20px 48px',
        }
      : {
          minHeight: '100vh',
          fontFamily,
          background:
            'linear-gradient(180deg, #0c0c0f 0%, #12121a 40%, #0f0f14 100%)',
          color: '#f1f5f9',
          padding: '32px 20px 48px',
        }

  return (
    <div style={bgStyle}>
      <div style={{ maxWidth: 960, margin: '0 auto' }}>
        <header
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 24,
            marginBottom: 36,
            padding: '28px 24px 32px',
            borderRadius: 24,
            background: theme === 'light'
              ? 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(241,245,253,0.95) 100%)'
              : 'linear-gradient(135deg, rgba(24,24,32,0.9) 0%, rgba(18,18,26,0.95) 100%)',
            border: theme === 'light' ? '1px solid rgba(99,102,241,0.12)' : '1px solid rgba(129,140,248,0.1)',
            boxShadow: theme === 'light'
              ? '0 4px 24px -4px rgba(99,102,241,0.15), 0 0 0 1px rgba(255,255,255,0.5) inset'
              : '0 4px 24px -4px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.03) inset',
          }}
        >
          <div style={{ flex: '1 1 320px' }}>
            <h1
              style={{
                fontSize: 30,
                fontWeight: 800,
                letterSpacing: '-0.03em',
                margin: 0,
                fontFamily,
                background:
                  theme === 'light'
                    ? 'linear-gradient(120deg, #4338ca 0%, #6366f1 40%, #7c3aed 100%)'
                    : 'linear-gradient(120deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%)',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
              }}
            >
              Ağ Anomali Tespiti
            </h1>
            <p style={{ color: colors.textMuted, marginTop: 10, fontSize: 16, maxWidth: 480, lineHeight: 1.6, fontWeight: 500 }}>
              Ağınızdaki trafiğin sakin mi, yoksa şüpheli mi olduğunu birkaç tıkla görebilirsiniz. Hazır modellerle deneme yapın, kendi trafik dosyalarınızı veya canlı trafiğinizi inceleyin.
            </p>
            <p style={{ color: colors.textMuted, marginTop: 8, fontSize: 14, maxWidth: 440, opacity: 0.95, fontWeight: 500 }}>
              &quot;Uygulamayı yükle&quot; ile telefon veya bilgisayara kurulur; masaüstü uygulaması gibi açılır.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div
              style={{
                padding: '10px 16px',
                borderRadius: 999,
                fontSize: 14,
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                color: colors.text,
                background: healthStatus === 'up' ? 'rgba(5,150,105,0.12)' : healthStatus === 'down' ? 'rgba(220,38,38,0.12)' : theme === 'light' ? 'rgba(100,116,139,0.12)' : 'rgba(148,163,184,0.1)',
                border: `1px solid ${healthStatus === 'up' ? 'rgba(5,150,105,0.35)' : healthStatus === 'down' ? 'rgba(220,38,38,0.35)' : theme === 'light' ? 'rgba(100,116,139,0.25)' : 'rgba(148,163,184,0.2)'}`,
              }}
            >
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: healthStatus === 'up' ? '#059669' : healthStatus === 'down' ? '#dc2626' : '#94a3b8', boxShadow: healthStatus === 'up' ? '0 0 8px rgba(5,150,105,0.5)' : 'none' }} />
              {healthStatus === 'up' ? 'Bağlı' : healthStatus === 'down' ? 'Bağlı değil' : '—'} {lastCheck && <span style={{ fontWeight: 500, opacity: 0.85 }}>· {lastCheck}</span>}
            </div>
            <button
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              style={{
                padding: '10px 18px',
                borderRadius: 999,
                border: theme === 'light' ? '1px solid rgba(148,163,184,0.4)' : '1px solid rgba(148,163,184,0.25)',
                background: theme === 'light' ? '#fff' : 'rgba(30,30,40,0.8)',
                color: colors.text,
                fontSize: 14,
                fontWeight: 600,
                cursor: 'pointer',
                boxShadow: theme === 'light' ? '0 1px 3px rgba(0,0,0,0.06)' : 'none',
              }}
            >
              {theme === 'light' ? 'Koyu mod' : 'Açık mod'}
            </button>
            {installable && (
              <button
                onClick={handleInstall}
                style={{
                  padding: '12px 20px',
                  borderRadius: 999,
                  border: 'none',
                  background: 'linear-gradient(135deg, #6366f1, #8b5cf6)',
                  color: 'white',
                  fontWeight: 700,
                  fontSize: 14,
                  cursor: 'pointer',
                  boxShadow: '0 4px 16px rgba(99,102,241,0.4)',
                }}
              >
                Uygulamayı yükle
              </button>
            )}
          </div>
        </header>

        {!API_BASE && !isLocal && (
          <p style={{ color: theme === 'light' ? '#b45309' : '#fbbf24', fontSize: 15, marginBottom: 20, padding: '12px 16px', borderRadius: 12, background: theme === 'light' ? 'rgba(245,158,11,0.1)' : 'rgba(251,191,36,0.1)', border: theme === 'light' ? '1px solid rgba(245,158,11,0.3)' : '1px solid rgba(251,191,36,0.3)', fontWeight: 500 }}>
            Canlıda backend adresi ayarlı değil. Vercel → Environment Variables → VITE_API_URL ekleyip yeniden deploy edin.
          </p>
        )}

        {/* 1. Bağlantı ve örnek analiz */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>Bağlantı ve örnek analiz</h2>
          <p style={makeDescStyle(theme)}>
            Sunucunun ayakta olduğunu kontrol edin, ardından seçtiğiniz modelin örnek trafik üzerinde verdiği risk skorunu görün.
          </p>
          <button type="button" onClick={() => setShowHelp((h) => !h)} style={{ marginBottom: 16, padding: '10px 16px', fontSize: 14, background: theme === 'light' ? 'rgba(99,102,241,0.08)' : 'rgba(129,140,248,0.1)', border: `1px solid ${colors.accent}33`, borderRadius: 12, color: colors.accent, cursor: 'pointer', fontWeight: 600 }}>
            {showHelp ? 'Kapat' : 'Ne işe yarar?'}
          </button>
          {showHelp && (
            <div style={{ marginBottom: 18, padding: 20, background: colors.boxBg, borderRadius: 16, border: `1px solid ${colors.inputBorder}`, fontSize: 15, color: colors.text, lineHeight: 1.65, fontWeight: 500 }}>
              <p style={{ margin: '0 0 10px 0' }}><strong>Bağlantıyı kontrol et:</strong> Sunucuya istek gider; &quot;çalışıyorum&quot; yanıtı gelirse bağlantı sağlam demektir.</p>
              <p style={{ margin: 0 }}><strong>Örnek analiz dene:</strong> Sabit örnek veri modele gönderilir; dönen risk skoru modelin canlı olduğunu gösterir.</p>
            </div>
          )}
          <div style={{ display: 'flex', gap: 20, marginBottom: 16, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <div>
              <button onClick={checkHealth} disabled={isChecking} style={{ minWidth: 200, padding: '12px 22px', borderRadius: 14, border: 'none', background: 'linear-gradient(135deg, #0ea5e9, #0284c7)', color: 'white', cursor: isChecking ? 'default' : 'pointer', fontWeight: 600, fontSize: 15, boxShadow: '0 4px 14px rgba(14,165,233,0.35)' }}>
                {isChecking ? 'Kontrol ediliyor...' : 'Bağlantıyı kontrol et'}
              </button>
              <p style={{ margin: '8px 0 0 0', fontSize: 13, color: colors.textMuted, fontWeight: 600, maxWidth: 220 }}>Yanıt &quot;ok&quot; ise API ayakta.</p>
            </div>
            <div>
              <button onClick={testScore} disabled={isScoring} style={{ minWidth: 200, padding: '12px 22px', borderRadius: 14, border: 'none', background: 'linear-gradient(135deg, #059669, #047857)', color: 'white', cursor: isScoring ? 'default' : 'pointer', fontWeight: 600, fontSize: 15, boxShadow: '0 4px 14px rgba(5,150,105,0.35)' }}>
                {isScoring ? 'Analiz ediliyor...' : 'Örnek analiz dene'}
              </button>
              <p style={{ margin: '8px 0 0 0', fontSize: 13, color: colors.textMuted, fontWeight: 600, maxWidth: 220 }}>Skor modelin canlı olduğunu gösterir.</p>
            </div>
            <div style={{ marginLeft: 4 }}>
              <span style={{ fontSize: 14, color: colors.textMuted, marginRight: 8, fontWeight: 600 }}>Arka plan:</span>
              <select
                value={backgroundMode}
                onChange={(e) => setBackgroundMode(e.target.value as 'off' | 'interval' | 'hours' | 'always')}
                style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 10, padding: '8px 12px', fontSize: 14, fontWeight: 500 }}
              >
                <option value="off">Kapalı</option>
                <option value="interval">Her X saatte bir / günde bir</option>
                <option value="hours">Sadece belirli saatlerde</option>
                <option value="always">Hep açık (sürekli takip)</option>
              </select>
              {backgroundMode === 'interval' && (
                <span style={{ marginLeft: 8 }}>
                  <select
                    value={checkIntervalMinutes}
                    onChange={(e) => setCheckIntervalMinutes(Number(e.target.value))}
                    style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '4px 8px', fontSize: 13 }}
                  >
                    <option value={60}>1 saatte bir</option>
                    <option value={180}>3 saatte bir</option>
                    <option value={300}>5 saatte bir</option>
                    <option value={600}>10 saatte bir</option>
                    <option value={1440}>1 günde bir</option>
                  </select>
                </span>
              )}
              {backgroundMode === 'hours' && (
                <span style={{ marginLeft: 8, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <label style={{ fontSize: 15, color: colors.textMuted }}>Saat</label>
                  <select value={hourStart} onChange={(e) => setHourStart(Number(e.target.value))} style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '4px 6px', fontSize: 13 }}>
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                    ))}
                  </select>
                  <span style={{ color: colors.textMuted }}>–</span>
                  <select value={hourEnd} onChange={(e) => setHourEnd(Number(e.target.value))} style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '4px 6px', fontSize: 13 }}>
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                    ))}
                  </select>
                </span>
              )}
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 14, color: colors.textMuted, fontWeight: 600 }}>Model:</span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ background: colors.inputBg, color: colors.text, borderRadius: 12, border: `1px solid ${colors.inputBorder}`, padding: '10px 14px', fontSize: 14, fontWeight: 600 }}
              >
                <option value="auto">Otomatik (önerilen)</option>
                <option value="ensemble">Ensemble</option>
                <option value="standard_ae">Standard Autoencoder</option>
                <option value="isolation_forest">Isolation Forest</option>
                <option value="one_class_svm">One-Class SVM</option>
                <option value="lstm_ae">LSTM Autoencoder</option>
              </select>
            </div>
          </div>
          {health && (
            <div style={{ marginTop: 18 }}>
              <p style={{ margin: '0 0 8px 0', fontSize: 13, color: colors.textMuted, fontWeight: 600 }}>Sunucu yanıtı</p>
              <pre style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: 16, borderRadius: 14, border: `1px solid ${colors.inputBorder}`, fontSize: 14, maxHeight: 160, overflow: 'auto', margin: 0, fontFamily: 'ui-monospace, monospace' }}>
                {health}
              </pre>
            </div>
          )}
          {scores !== null && (
            <div style={{ marginTop: 20, padding: 20, borderRadius: 18, background: theme === 'light' ? 'linear-gradient(135deg, rgba(99,102,241,0.08), rgba(139,92,246,0.06))' : 'linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08))', border: theme === 'light' ? '1px solid rgba(99,102,241,0.2)' : '1px solid rgba(129,140,248,0.2)', borderLeft: `4px solid ${colors.accent}` }}>
              <p style={{ margin: '0 0 8px 0', fontSize: 13, color: colors.textMuted, fontWeight: 600 }}>Örnek analiz sonucu</p>
              <p style={{ margin: 0, fontSize: 16, color: colors.text, fontWeight: 600 }}>
                Risk skoru: {scores.map((x) => (typeof x === 'number' ? x.toFixed(4) : x)).join(', ')}
                {usedModelName && <span style={{ color: colors.accent, marginLeft: 10, fontWeight: 500 }}>({modelLabels[usedModelName] || usedModelName})</span>}
              </p>
              {scoreNote && <p style={{ fontSize: 15, color: colors.text, marginTop: 10, marginBottom: 0, fontWeight: 500 }}>{scoreNote}</p>}
            </div>
          )}
        </section>

        {/* 2. CSV ile toplu analiz */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>CSV ile toplu analiz</h2>
          <p style={makeDescStyle(theme)}>
            Özellik kolonlarına uyumlu CSV yükleyin; model tüm satırları tarayıp anomali oranı ve skor özetini verir.
          </p>
          <p style={{ fontSize: 14, color: colors.textMuted, marginTop: -4, marginBottom: 14, fontWeight: 500 }}>Dosyalar yalnızca analiz için kullanılır, saklanmaz. En fazla 50 MB.</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center', marginBottom: 14 }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', padding: '10px 18px', borderRadius: 14, background: colors.inputBg, border: `1px solid ${colors.inputBorder}`, cursor: 'pointer', fontWeight: 600, fontSize: 14, color: colors.accent }}>
              <input ref={csvInputRef} type="file" accept=".csv" style={{ display: 'none' }} />
              Dosya seç
            </label>
            <button
              onClick={analyzeCsv}
              disabled={csvLoading}
              style={{
                padding: '12px 22px',
                borderRadius: 14,
                border: 'none',
                background: 'linear-gradient(135deg, #6366f1, #4f46e5)',
                color: 'white',
                cursor: csvLoading ? 'default' : 'pointer',
                fontSize: 15,
                fontWeight: 600,
                boxShadow: '0 4px 16px rgba(99,102,241,0.35)',
              }}
            >
              {csvLoading ? 'Analiz ediliyor...' : 'CSV analiz et'}
            </button>
          </div>
          {csvError && (
            <p style={{ color: colors.danger, fontSize: 14, marginTop: 8, padding: '10px 14px', borderRadius: 12, background: theme === 'light' ? 'rgba(220,38,38,0.08)' : 'rgba(220,38,38,0.15)', fontWeight: 500 }}>{csvError}</p>
          )}
          {csvSummary && (
            <div style={{ marginTop: 16, padding: 20, borderRadius: 16, background: colors.boxBg, border: `1px solid ${colors.inputBorder}`, borderLeft: `4px solid ${colors.accent}`, fontSize: 15, color: colors.text, fontWeight: 500 }}>
              <p style={{ margin: '0 0 6px 0' }}><strong>Model:</strong> {csvSummary.model in modelLabels ? modelLabels[csvSummary.model] : csvSummary.model}</p>
              <p style={{ margin: '0 0 6px 0' }}><strong>Toplam satır:</strong> {csvSummary.total_rows}</p>
              <p style={{ margin: '0 0 6px 0' }}><strong>Anomali:</strong> {csvSummary.anomaly_count} ({(csvSummary.anomaly_ratio * 100).toFixed(2)}%)</p>
              <p style={{ margin: 0 }}><strong>Skor:</strong> min {csvSummary.score_min.toFixed(4)} · ort {csvSummary.score_mean.toFixed(4)} · max {csvSummary.score_max.toFixed(4)}</p>
            </div>
          )}
        </section>

        {/* 3. PCAP ile toplu analiz */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>PCAP ile toplu analiz</h2>
          <p style={makeDescStyle(theme)}>
            Wireshark veya benzeri araçlarla kaydettiğiniz `.pcap` / `.pcapng` dosyasını yükleyin; akış bazında anomali oranı ve skor özeti hesaplanır.
          </p>
          <p style={{ fontSize: 14, color: colors.textMuted, marginTop: -4, marginBottom: 14, fontWeight: 500 }}>Dosyalar yalnızca analiz için kullanılır, saklanmaz. En fazla 50 MB.</p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14, alignItems: 'center', marginBottom: 14 }}>
            <label style={{ display: 'inline-flex', alignItems: 'center', padding: '10px 18px', borderRadius: 14, background: colors.inputBg, border: `1px solid ${colors.inputBorder}`, cursor: 'pointer', fontWeight: 600, fontSize: 14, color: '#0ea5e9' }}>
              <input ref={pcapInputRef} type="file" accept=".pcap,.pcapng" style={{ display: 'none' }} />
              PCAP seç
            </label>
            <button
              onClick={analyzePcap}
              disabled={pcapLoading}
              style={{
                padding: '12px 22px',
                borderRadius: 14,
                border: 'none',
                background: 'linear-gradient(135deg, #0ea5e9, #0284c7)',
                color: 'white',
                cursor: pcapLoading ? 'default' : 'pointer',
                fontSize: 15,
                fontWeight: 600,
                boxShadow: '0 4px 16px rgba(14,165,233,0.35)',
              }}
            >
              {pcapLoading ? 'Analiz ediliyor...' : 'PCAP analiz et'}
            </button>
          </div>
          {pcapError && (
            <p style={{ color: colors.danger, fontSize: 14, marginTop: 8, padding: '10px 14px', borderRadius: 12, background: theme === 'light' ? 'rgba(220,38,38,0.08)' : 'rgba(220,38,38,0.15)', fontWeight: 500 }}>{pcapError}</p>
          )}
          {pcapSummary && (
            <div style={{ marginTop: 16, padding: 20, borderRadius: 16, background: colors.boxBg, border: `1px solid ${colors.inputBorder}`, borderLeft: '4px solid #0ea5e9', fontSize: 15, color: colors.text, fontWeight: 500 }}>
              <p style={{ margin: '0 0 6px 0' }}><strong>Model:</strong> {pcapSummary.model in modelLabels ? modelLabels[pcapSummary.model] : pcapSummary.model}</p>
              <p style={{ margin: '0 0 6px 0' }}><strong>Toplam flow:</strong> {pcapSummary.total_rows}</p>
              <p style={{ margin: '0 0 6px 0' }}><strong>Anomali:</strong> {pcapSummary.anomaly_count} ({(pcapSummary.anomaly_ratio * 100).toFixed(2)}%)</p>
              <p style={{ margin: 0 }}><strong>Skor:</strong> min {pcapSummary.score_min.toFixed(4)} · ort {pcapSummary.score_mean.toFixed(4)} · max {pcapSummary.score_max.toFixed(4)}</p>
            </div>
          )}
        </section>

        {/* 4. Kendi ağını analiz et */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>Kendi bilgisayarınızda canlı analiz</h2>
          <p style={makeDescStyle(theme)}>
            Sensörü çalıştırarak ağ trafiğinizi canlı izleyin. Şüpheli akışlar uyarı olarak gösterilir; durdurmak için Ctrl+C.
          </p>
          <div style={{ background: colors.boxBg, padding: 22, borderRadius: 18, border: `1px solid ${colors.inputBorder}`, borderLeft: `4px solid ${colors.accent}`, fontSize: 15, color: colors.text, lineHeight: 1.7, fontWeight: 500 }}>
            <p style={{ margin: '0 0 12px 0', fontWeight: 700, fontSize: 16, color: colors.accent }}>Nasıl yapılır?</p>
            <ol style={{ margin: 0, paddingLeft: 22 }}>
              <li>PowerShell veya terminali <strong>Yönetici olarak</strong> açın.</li>
              <li>Proje klasöründe sanal ortamı açın: <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '4px 10px', borderRadius: 8, fontSize: 14, fontFamily: 'ui-monospace, monospace' }}>.\venv\Scripts\activate</code></li>
              <li>Çalıştırın: <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '4px 10px', borderRadius: 8, fontSize: 14, fontFamily: 'ui-monospace, monospace' }}>python realtime_nids_scapy.py</code></li>
            </ol>
            <p style={{ margin: '16px 0 0 0', fontSize: 14, color: colors.textMuted }}>
              Normal trafikte az uyarı, sıra dışı akışlarda daha fazla uyarı görürsünüz.
            </p>
          </div>
        </section>

        {/* 5. Model karşılaştırma */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>Modeller ve performans</h2>
          <p style={makeDescStyle(theme)}>
            Sunucuda yüklü modeller ve (karşılaştırma çalıştırıldıysa) doğruluk, F1, hassasiyet gibi metrikler aşağıda.
          </p>
          {apiModels && (
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center', marginBottom: 16 }}>
              <span style={{ color: colors.accent, fontSize: 14, fontWeight: 700 }}>Yüklü: {apiModels.models.length ? apiModels.models.map((m) => modelLabels[m] || m).join(', ') : 'yok'}</span>
              {apiModels.ensemble_available && <span style={{ padding: '6px 14px', borderRadius: 999, background: theme === 'light' ? 'rgba(99,102,241,0.12)' : 'rgba(129,140,248,0.2)', border: `1px solid ${colors.accent}55`, fontSize: 14, fontWeight: 700, color: colors.accent }}>Ensemble kullanılabilir</span>}
            </div>
          )}
          {comparison?.available && comparison.rows?.length > 0 ? (
            <>
              <div style={{ marginBottom: 20, padding: 20, borderRadius: 18, background: colors.boxBg, border: `1px solid ${colors.inputBorder}` }}>
                <p style={{ margin: '0 0 14px 0', fontSize: 15, fontWeight: 700, color: colors.accent }}>F1 skoru karşılaştırması</p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                  {comparison.rows.map((row: Record<string, unknown>, i: number) => {
                    const f1 = typeof row.f1 === 'number' ? row.f1 : 0;
                    const pct = Math.min(100, Math.round(f1 * 100));
                    return (
                      <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                        <span style={{ width: 160, fontSize: 14, fontWeight: 600, color: colors.text, flexShrink: 0 }}>{modelLabels[String(row.model)] || String(row.model)}</span>
                        <div style={{ flex: 1, height: 24, background: theme === 'light' ? '#e2e8f0' : 'rgba(51,65,85,0.6)', borderRadius: 12, overflow: 'hidden' }}>
                          <div style={{ width: `${pct}%`, height: '100%', background: theme === 'light' ? 'linear-gradient(90deg, #6366f1, #8b5cf6)' : 'linear-gradient(90deg, #818cf8, #a78bfa)', borderRadius: 12, transition: 'width 0.4s ease' }} />
                        </div>
                        <span style={{ fontSize: 14, fontWeight: 700, color: colors.text, width: 48, textAlign: 'right' }}>{(f1 * 100).toFixed(1)}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            <div style={{ overflowX: 'auto', borderRadius: 18, border: `1px solid ${colors.inputBorder}`, background: colors.boxBg, marginBottom: 18 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 14 }}>
                <thead>
                  <tr style={{ background: theme === 'light' ? 'rgba(99,102,241,0.08)' : 'rgba(99,102,241,0.12)' }}>
                    <th style={{ textAlign: 'left', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>Model</th>
                    <th style={{ textAlign: 'right', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>Accuracy</th>
                    <th style={{ textAlign: 'right', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>F1</th>
                    <th style={{ textAlign: 'right', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>Recall</th>
                    <th style={{ textAlign: 'right', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>Precision</th>
                    <th style={{ textAlign: 'right', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>ROC-AUC</th>
                    <th style={{ textAlign: 'right', padding: '14px 18px', color: colors.accent, fontWeight: 700 }}>FPR</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row: Record<string, unknown>, i: number) => (
                    <tr key={i} style={{ background: i % 2 === 1 ? (theme === 'light' ? 'rgba(248,250,252,0.8)' : 'rgba(255,255,255,0.02)') : 'transparent' }}>
                      <td style={{ padding: '12px 18px', color: colors.text, fontWeight: 600 }}>{modelLabels[String(row.model)] || String(row.model)}</td>
                      <td style={{ padding: '12px 18px', textAlign: 'right', color: colors.textMuted }}>{typeof row.accuracy === 'number' ? (row.accuracy as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '12px 18px', textAlign: 'right', color: colors.textMuted }}>{typeof row.f1 === 'number' ? (row.f1 as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '12px 18px', textAlign: 'right', color: colors.textMuted }}>{typeof row.recall === 'number' ? (row.recall as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '12px 18px', textAlign: 'right', color: colors.textMuted }}>{typeof row.precision === 'number' ? (row.precision as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '12px 18px', textAlign: 'right', color: colors.textMuted }}>{typeof row.roc_auc === 'number' ? (row.roc_auc as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '12px 18px', textAlign: 'right', color: colors.textMuted }}>{typeof row.false_positive_rate === 'number' ? (row.false_positive_rate as number).toFixed(4) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            </>
          ) : (
            <div style={{ background: colors.boxBg, padding: 18, borderRadius: 16, border: `1px solid ${colors.inputBorder}`, fontSize: 15, color: colors.textMuted, marginBottom: 18, fontWeight: 500 }}>
              Karşılaştırma verisi yok. Lokalde: <code style={{ display: 'block', marginTop: 8, background: colors.codeSoftBg, color: colors.codeSoftText, padding: '10px 14px', borderRadius: 10, fontFamily: 'ui-monospace, monospace' }}>python scripts\experiments\compare_core_models.py</code>
              Sonra backend&#39;i yeniden başlatıp sayfayı yenileyin.
            </div>
          )}
          <div style={{ marginBottom: 18, padding: 18, borderRadius: 16, background: theme === 'light' ? 'rgba(99,102,241,0.06)' : 'rgba(99,102,241,0.1)', border: `1px solid ${colors.accent}33`, fontSize: 15, color: colors.text, lineHeight: 1.6, fontWeight: 500 }}>
            <strong style={{ color: colors.accent }}>Metrikleri nasıl okursunuz?</strong> F1 ve Accuracy yüksek (örn. &gt;0,7), FPR düşükse model güvenilir. ROC-AUC 1&#39;e yakınsa anomali ile normal trafiği iyi ayırıyor demektir.
          </div>
          <div style={{ overflowX: 'auto', borderRadius: 16, border: `2px solid ${colors.inputBorder}`, background: colors.boxBg }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 15 }}>
              <thead>
                <tr style={{ background: theme === 'light' ? 'rgba(99,102,241,0.12)' : 'rgba(99,102,241,0.15)', borderBottom: `2px solid ${colors.inputBorder}` }}>
                  <th style={{ textAlign: 'left', padding: '14px 18px', color: colors.accent, fontWeight: 800, fontSize: 15 }}>Model</th>
                  <th style={{ textAlign: 'left', padding: '14px 18px', color: colors.accent, fontWeight: 800, fontSize: 15 }}>Sunucu durumu</th>
                </tr>
              </thead>
              <tbody>
                {['standard_ae', 'isolation_forest', 'one_class_svm', 'lstm_ae', 'ensemble'].map((key, i) => (
                  <tr key={key} style={{ background: i % 2 === 1 ? (theme === 'light' ? 'rgba(248,250,252,0.8)' : 'rgba(255,255,255,0.03)') : 'transparent', borderBottom: `1px solid ${colors.inputBorder}` }}>
                    <td style={{ padding: '14px 18px', color: colors.text, fontWeight: 700, fontSize: 15 }}>{modelLabels[key] || key}</td>
                    <td style={{ padding: '14px 18px', fontSize: 15, fontWeight: 600 }}>
                      {key === 'ensemble' ? (
                        apiModels?.ensemble_available ? <span style={{ color: colors.success }}>Kullanılabilir</span> : <span style={{ color: colors.textMuted }}>En az bir model gerekli</span>
                      ) : (
                        apiModels?.models?.includes(key) ? <span style={{ color: colors.success }}>Yüklü</span> : <span style={{ color: colors.textMuted }}>Yok</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Sona: Nasıl indirilir ve çalıştırılır */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>Nasıl indirilir ve çalıştırılır?</h2>
          <p style={makeDescStyle(theme)}>
            Projeyi bilgisayarınıza indirip yerelde çalıştırmak veya sadece bu sayfayı kullanmak için aşağıdaki adımları takip edin.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginTop: 8 }}>
            <div style={{ padding: 20, borderRadius: 16, background: theme === 'light' ? 'rgba(99,102,241,0.06)' : 'rgba(99,102,241,0.1)', border: `1px solid ${colors.accent}33`, borderLeft: `4px solid ${colors.accent}` }}>
              <p style={{ margin: '0 0 12px 0', fontWeight: 800, fontSize: 16, color: colors.accent }}>İndirme</p>
              <ol style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8, fontSize: 14, color: colors.text, fontWeight: 600 }}>
                <li><strong>Git:</strong> <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>git clone https://github.com/Ebrarbulut/b2.git</code></li>
                <li><strong>ZIP:</strong> GitHub sayfasında Code → Download ZIP, açıp klasöre girin.</li>
                <li>İndirdikten sonra klasör içinde terminal açın (örn. <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>cd b2</code>).</li>
              </ol>
            </div>
            <div style={{ padding: 20, borderRadius: 16, background: theme === 'light' ? 'rgba(5,150,105,0.06)' : 'rgba(5,150,105,0.1)', border: `1px solid ${colors.success}44`, borderLeft: `4px solid ${colors.success}` }}>
              <p style={{ margin: '0 0 12px 0', fontWeight: 800, fontSize: 16, color: colors.success }}>Çalıştırma</p>
              <ol style={{ margin: 0, paddingLeft: 20, lineHeight: 1.8, fontSize: 14, color: colors.text, fontWeight: 600 }}>
                <li><strong>Backend:</strong> <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>cd backend</code> → <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>pip install -r requirements.txt</code> → <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>uvicorn main:app --host 0.0.0.0 --port 8000</code></li>
                <li><strong>Frontend:</strong> Yeni terminalde <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>cd frontend</code> → <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>npm install</code> → <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>npm run dev</code></li>
                <li>Tarayıcıda <strong>http://localhost:5173</strong> açın. Backend 8000 portunda çalışıyor olmalı.</li>
              </ol>
            </div>
            <div style={{ padding: 20, borderRadius: 16, background: theme === 'light' ? 'rgba(14,165,233,0.06)' : 'rgba(14,165,233,0.1)', border: '1px solid rgba(14,165,233,0.3)', borderLeft: '4px solid #0ea5e9' }}>
              <p style={{ margin: '0 0 12px 0', fontWeight: 800, fontSize: 16, color: '#0ea5e9' }}>Canlı NIDS (.exe)</p>
              <p style={{ margin: 0, fontSize: 14, color: colors.text, lineHeight: 1.7, fontWeight: 600 }}>
                Proje klasöründe <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>scripts\build_nad_sensor_exe.bat</code> çalıştırın. Oluşan <code style={{ background: colors.codeSoftBg, color: colors.codeSoftText, padding: '3px 8px', borderRadius: 8, fontSize: 13 }}>dist\nad_sensor.exe</code> dosyasına çift tıklayarak sensörü (terminal açılmadan) çalıştırabilirsiniz. Yönetici izni gerekebilir.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  )
}
