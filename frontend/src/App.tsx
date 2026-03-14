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

const makeCardStyle = (theme: 'light' | 'dark') =>
  ({
    marginTop: 24,
    padding: '24px 22px',
    borderRadius: 18,
    background:
      theme === 'light'
        ? 'linear-gradient(145deg, #ffffff 0%, #fdf2f8 30%, #eff6ff 100%)'
        : 'linear-gradient(145deg, rgba(20,20,20,0.95), rgba(30,30,35,0.9))',
    border:
      theme === 'light'
        ? '1px solid rgba(219,39,119,0.25)'
        : '1px solid rgba(244,114,182,0.2)',
    boxShadow:
      theme === 'light'
        ? '0 4px 24px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.04)'
        : '0 4px 24px rgba(0,0,0,0.4)',
  } as const)

const makeTitleStyle = (theme: 'light' | 'dark') =>
  ({
    margin: '0 0 6px 0',
    fontSize: 17,
    fontWeight: 600,
    background:
      theme === 'light'
        ? 'linear-gradient(90deg, #db2777, #4f46e5)'
        : 'linear-gradient(90deg, #f9a8d4, #93c5fd)',
    WebkitBackgroundClip: 'text' as const,
    color: 'transparent',
  })

const makeDescStyle = (theme: 'light' | 'dark') =>
  ({
    color: theme === 'light' ? '#374151' : '#d1d5db',
    fontSize: 13,
    marginBottom: 14,
    lineHeight: 1.55,
  } as const)

// Tema renkleri: okunabilirlik ve premium his için yüksek kontrast
const themeColors = (theme: 'light' | 'dark') => ({
  text: theme === 'light' ? '#111827' : '#f3f4f6',
  textMuted: theme === 'light' ? '#4b5563' : '#9ca3af',
  cardBg: theme === 'light' ? 'rgba(255,255,255,0.85)' : 'rgba(17,17,17,0.6)',
  inputBg: theme === 'light' ? '#f9fafb' : 'rgba(30,30,30,0.9)',
  inputBorder: theme === 'light' ? 'rgba(107,114,128,0.4)' : 'rgba(148,163,184,0.4)',
  boxBg: theme === 'light' ? '#f3f4f6' : 'rgba(30,30,30,0.8)',
  preBg: theme === 'light' ? '#1f2937' : '#0f172a',
  preText: theme === 'light' ? '#e5e7eb' : '#cbd5e1',
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
          background:
            'radial-gradient(circle at top, #ffe4f3 0, #faf5f9 40%, #eff6ff 70%, #e0f2fe 100%)',
          color: '#111827',
          padding: '20px 16px 40px',
        }
      : {
          minHeight: '100vh',
          background: '#0a0a0a',
          color: '#f3f4f6',
          padding: '20px 16px 40px',
        }

  return (
    <div
      style={bgStyle}
    >
      <div style={{ maxWidth: 920, margin: '0 auto' }}>
        {/* Üst: Başlık + API durumu + Yükle */}
        <header
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 16,
            marginBottom: 28,
          }}
        >
          <div>
            <h1
              style={{
                fontSize: 24,
                fontWeight: 700,
                margin: 0,
                background:
                  theme === 'light'
                    ? 'linear-gradient(120deg, #db2777, #4f46e5)'
                    : 'linear-gradient(120deg, #f9a8d4, #93c5fd)',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
              }}
            >
              Ağ Anomali Tespiti
            </h1>
            <p style={{ color: colors.textMuted, marginTop: 6, fontSize: 14, maxWidth: 460, lineHeight: 1.5 }}>
              Ağınızdaki trafiğin sakin mi, yoksa şüpheli mi olduğunu birkaç tıkla görebilirsiniz. İsterseniz hazır modellerle deneme yapın, isterseniz kendi trafik dosyalarınızı veya canlı trafiğinizi inceleyin.
            </p>
            <p style={{ color: colors.textMuted, marginTop: 6, fontSize: 12, maxWidth: 420, opacity: 0.9 }}>
              Uygulamayı tarayıcıdan &quot;Uygulamayı yükle&quot; diyerek telefonunuza ya da bilgisayarınıza kurabilirsiniz. Kurulduktan sonra normal bir uygulama gibi açılır (masaüstü .exe yerine modern web uygulaması olarak çalışır).
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div
              style={{
                padding: '8px 14px',
                borderRadius: 12,
                background: healthStatus === 'up' ? 'rgba(34,197,94,0.15)' : healthStatus === 'down' ? 'rgba(239,68,68,0.15)' : theme === 'light' ? 'rgba(107,114,128,0.15)' : 'rgba(148,163,184,0.15)',
                border: `1px solid ${healthStatus === 'up' ? 'rgba(34,197,94,0.5)' : healthStatus === 'down' ? 'rgba(239,68,68,0.5)' : theme === 'light' ? 'rgba(107,114,128,0.4)' : 'rgba(148,163,184,0.4)'}`,
                fontSize: 13,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                color: colors.text,
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: healthStatus === 'up' ? '#22c55e' : healthStatus === 'down' ? '#ef4444' : '#94a3b8' }} />
              Sunucu: {healthStatus === 'up' ? 'Bağlı' : healthStatus === 'down' ? 'Bağlı değil' : '—'} {lastCheck && ` · ${lastCheck}`}
            </div>
            <button
              onClick={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
              style={{
                padding: '8px 14px',
                borderRadius: 999,
                border: theme === 'light' ? '1px solid #d1d5db' : '1px solid rgba(148,163,184,0.5)',
                background: theme === 'light' ? '#fff' : 'rgba(30,30,30,0.8)',
                color: colors.text,
                fontSize: 12,
                fontWeight: 500,
                cursor: 'pointer',
              }}
            >
              {theme === 'light' ? 'Koyu moda geç' : 'Açık moda geç'}
            </button>
            {installable && (
              <button
                onClick={handleInstall}
                style={{
                  padding: '10px 18px',
                  borderRadius: 12,
                  border: 'none',
                  background: 'linear-gradient(135deg, #a855f7, #6366f1)',
                  color: 'white',
                  fontWeight: 600,
                  fontSize: 14,
                  cursor: 'pointer',
                  boxShadow: '0 4px 14px rgba(168,85,247,0.4)',
                }}
              >
                Uygulamayı yükle
              </button>
            )}
          </div>
        </header>

        {!API_BASE && !isLocal && (
          <p style={{ color: theme === 'light' ? '#b45309' : '#fbbf24', fontSize: 13, marginBottom: 16 }}>
            Canlıda backend adresi ayarlı değil. Vercel → Environment Variables → VITE_API_URL ekleyip yeniden deploy edin.
          </p>
        )}

        {/* 1. Bağlantı ve örnek analiz */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>Bağlantı ve örnek analiz</h2>
          <p style={makeDescStyle(theme)}>
            Önce sunucunun ayakta olduğunu kontrol edin, ardından seçtiğiniz modelin örnek bir trafik üzerinde verdiği risk skorunu görün.
          </p>
          <button type="button" onClick={() => setShowHelp((h) => !h)} style={{ marginBottom: 12, padding: '8px 14px', fontSize: 12, background: colors.boxBg, border: `1px solid ${colors.inputBorder}`, borderRadius: 10, color: theme === 'light' ? '#6366f1' : '#93c5fd', cursor: 'pointer', fontWeight: 500 }}>
            {showHelp ? 'Ne işe yarar? (kapat)' : 'Ne işe yarar?'}
          </button>
          {showHelp && (
            <div style={{ marginBottom: 14, padding: 16, background: colors.boxBg, borderRadius: 12, border: `1px solid ${colors.inputBorder}`, fontSize: 13, color: colors.text, lineHeight: 1.6 }}>
              <p style={{ margin: '0 0 8px 0' }}><strong>Bağlantıyı kontrol et:</strong> Sunucuya bir istek gider; &quot;çalışıyorum&quot; yanıtı gelirse bağlantı sağlam demektir.</p>
              <p style={{ margin: 0 }}><strong>Örnek analiz dene:</strong> Sabit bir örnek veri modele gönderilir; dönen risk skoru modelin canlı ve çalışır durumda olduğunu gösterir.</p>
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, marginBottom: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <div>
              <button onClick={checkHealth} disabled={isChecking} style={{ padding: '10px 18px', borderRadius: 12, border: 'none', background: '#0ea5e9', color: 'white', cursor: isChecking ? 'default' : 'pointer', fontWeight: 500 }}>
                {isChecking ? 'Kontrol ediliyor...' : 'Bağlantıyı kontrol et'}
              </button>
              <p style={{ margin: '4px 0 0 0', fontSize: 11, color: colors.textMuted, maxWidth: 220 }}>Sunucuya istek atar; yanıt &quot;ok&quot; ise API ayakta.</p>
            </div>
            <div>
              <button onClick={testScore} disabled={isScoring} style={{ padding: '10px 18px', borderRadius: 12, border: 'none', background: '#22c55e', color: 'white', cursor: isScoring ? 'default' : 'pointer', fontWeight: 500 }}>
                {isScoring ? 'Analiz ediliyor...' : 'Örnek analiz dene'}
              </button>
              <p style={{ margin: '4px 0 0 0', fontSize: 11, color: colors.textMuted, maxWidth: 220 }}>Örnek veriyi modele gönderir; skor modelin canlı olduğunu gösterir.</p>
            </div>
            <div style={{ marginLeft: 4 }}>
              <span style={{ fontSize: 12, color: colors.textMuted, marginRight: 8 }}>Arka plan kontrolü:</span>
              <select
                value={backgroundMode}
                onChange={(e) => setBackgroundMode(e.target.value as 'off' | 'interval' | 'hours' | 'always')}
                style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '6px 10px', fontSize: 12 }}
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
                    style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '4px 8px', fontSize: 12 }}
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
                  <label style={{ fontSize: 12, color: colors.textMuted }}>Saat</label>
                  <select value={hourStart} onChange={(e) => setHourStart(Number(e.target.value))} style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '4px 6px', fontSize: 12 }}>
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                    ))}
                  </select>
                  <span style={{ color: colors.textMuted }}>–</span>
                  <select value={hourEnd} onChange={(e) => setHourEnd(Number(e.target.value))} style={{ background: colors.inputBg, color: colors.text, border: `1px solid ${colors.inputBorder}`, borderRadius: 8, padding: '4px 6px', fontSize: 12 }}>
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                    ))}
                  </select>
                </span>
              )}
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, color: colors.textMuted }}>Analizde kullanılacak model:</span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ background: colors.inputBg, color: colors.text, borderRadius: 12, border: `1px solid ${colors.inputBorder}`, padding: '8px 12px', fontSize: 13 }}
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
            <div style={{ marginTop: 12 }}>
              <p style={{ margin: '0 0 6px 0', fontSize: 12, color: colors.textMuted }}>Sunucu yanıtı</p>
              <pre style={{ background: colors.preBg, color: colors.preText, padding: 12, borderRadius: 12, border: `1px solid ${colors.inputBorder}`, fontSize: 12, maxHeight: 160, overflow: 'auto', margin: 0 }}>
                {health}
              </pre>
            </div>
          )}
          {scores !== null && (
            <div style={{ marginTop: 16, padding: 18, borderRadius: 14, background: theme === 'light' ? 'linear-gradient(135deg, rgba(251,207,232,0.4), rgba(219,234,254,0.4))' : 'linear-gradient(135deg, rgba(248,113,165,0.12), rgba(59,130,246,0.08))', border: theme === 'light' ? '1px solid rgba(219,39,119,0.3)' : '1px solid rgba(248,113,165,0.3)' }}>
              <p style={{ margin: '0 0 6px 0', fontSize: 12, color: colors.textMuted }}>Örnek analiz sonucu</p>
              <p style={{ margin: 0, fontSize: 14, color: colors.text }}>
                <strong>Risk skoru:</strong> {scores.map((x) => (typeof x === 'number' ? x.toFixed(4) : x)).join(', ')}
                {usedModelName && <span style={{ color: theme === 'light' ? '#6366f1' : '#93c5fd', marginLeft: 8 }}>(model: {modelLabels[usedModelName] || usedModelName})</span>}
              </p>
              {scoreNote && <p style={{ fontSize: 13, color: colors.text, marginTop: 8, marginBottom: 0 }}>{scoreNote}</p>}
            </div>
          )}
        </section>

        {/* 2. CSV ile toplu analiz */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>CSV ile toplu analiz</h2>
          <p style={makeDescStyle(theme)}>
            Eğitimde kullandığınız özellik kolonlarına uyumlu bir CSV yükleyin; seçtiğiniz model tüm satırları tarayıp anomali oranını ve skor özetini verir.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 10 }}>
            <input
              ref={csvInputRef}
              type="file"
              accept=".csv"
              style={{ fontSize: 12, color: colors.text }}
            />
            <button
              onClick={analyzeCsv}
              disabled={csvLoading}
              style={{
                padding: '8px 16px',
                borderRadius: 12,
                border: 'none',
                background: '#6366f1',
                color: 'white',
                cursor: csvLoading ? 'default' : 'pointer',
                fontSize: 13,
                fontWeight: 500,
              }}
            >
              {csvLoading ? 'Analiz ediliyor...' : 'CSV dosyasını analiz et'}
            </button>
          </div>
          {csvError && (
            <p style={{ color: '#f97373', fontSize: 12, marginTop: 4 }}>{csvError}</p>
          )}
          {csvSummary && (
            <div style={{ marginTop: 10, padding: 14, borderRadius: 12, background: colors.boxBg, border: `1px solid ${colors.inputBorder}`, fontSize: 13, color: colors.text }}>
              <p style={{ margin: '0 0 4px 0' }}>
                <strong>Model:</strong> {csvSummary.model in modelLabels ? modelLabels[csvSummary.model] : csvSummary.model}
              </p>
              <p style={{ margin: '0 0 4px 0' }}>
                <strong>Toplam satır:</strong> {csvSummary.total_rows}
              </p>
              <p style={{ margin: '0 0 4px 0' }}>
                <strong>Anomali sayısı:</strong> {csvSummary.anomaly_count} ({(csvSummary.anomaly_ratio * 100).toFixed(2)}%)
              </p>
              <p style={{ margin: 0 }}>
                <strong>Skor aralığı:</strong> min {csvSummary.score_min.toFixed(4)} · ort {csvSummary.score_mean.toFixed(4)} · max {csvSummary.score_max.toFixed(4)}
              </p>
            </div>
          )}
        </section>

        {/* 3. PCAP ile toplu analiz */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>PCAP ile toplu analiz</h2>
          <p style={makeDescStyle(theme)}>
            Wireshark veya benzeri araçlarla kaydettiğiniz `.pcap` / `.pcapng` dosyasını yükleyin; akış bazında anomali oranı ve skor özeti hesaplanır.
          </p>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10, alignItems: 'center', marginBottom: 10 }}>
            <input
              ref={pcapInputRef}
              type="file"
              accept=".pcap,.pcapng"
              style={{ fontSize: 12, color: colors.text }}
            />
            <button
              onClick={analyzePcap}
              disabled={pcapLoading}
              style={{
                padding: '8px 16px',
                borderRadius: 12,
                border: 'none',
                background: '#0ea5e9',
                color: 'white',
                cursor: pcapLoading ? 'default' : 'pointer',
                fontSize: 13,
                fontWeight: 500,
              }}
            >
              {pcapLoading ? 'Analiz ediliyor...' : 'PCAP dosyasını analiz et'}
            </button>
          </div>
          {pcapError && (
            <p style={{ color: '#f97373', fontSize: 12, marginTop: 4 }}>{pcapError}</p>
          )}
          {pcapSummary && (
            <div style={{ marginTop: 10, padding: 14, borderRadius: 12, background: colors.boxBg, border: `1px solid ${colors.inputBorder}`, fontSize: 13, color: colors.text }}>
              <p style={{ margin: '0 0 4px 0' }}>
                <strong>Model:</strong> {pcapSummary.model in modelLabels ? modelLabels[pcapSummary.model] : pcapSummary.model}
              </p>
              <p style={{ margin: '0 0 4px 0' }}>
                <strong>Toplam flow:</strong> {pcapSummary.total_rows}
              </p>
              <p style={{ margin: '0 0 4px 0' }}>
                <strong>Anomali sayısı:</strong> {pcapSummary.anomaly_count} ({(pcapSummary.anomaly_ratio * 100).toFixed(2)}%)
              </p>
              <p style={{ margin: 0 }}>
                <strong>Skor aralığı:</strong> min {pcapSummary.score_min.toFixed(4)} · ort {pcapSummary.score_mean.toFixed(4)} · max {pcapSummary.score_max.toFixed(4)}
              </p>
            </div>
          )}
        </section>

        {/* 4. Kendi ağını analiz et */}
        <section style={makeCardStyle(theme)}>
          <h2 style={makeTitleStyle(theme)}>Kendi bilgisayarınızda canlı analiz</h2>
          <p style={makeDescStyle(theme)}>
            Kendi bilgisayarınızda sensörü çalıştırarak ağ trafiğinizi canlı izleyebilirsiniz. Şüpheli akışlar konsolda uyarı olarak gösterilir; durdurmak için Ctrl+C yeterli.
          </p>
          <div style={{ background: colors.boxBg, padding: 18, borderRadius: 14, border: `1px solid ${colors.inputBorder}`, fontSize: 13, color: colors.text, lineHeight: 1.7 }}>
            <p style={{ margin: '0 0 10px 0', fontWeight: 600 }}>Nasıl yapılır?</p>
            <ol style={{ margin: 0, paddingLeft: 20 }}>
              <li>PowerShell veya terminali <strong>Yönetici olarak</strong> açın.</li>
              <li>Proje klasörüne gidip sanal ortamı açın: <code style={{ background: colors.preBg, color: colors.preText, padding: '2px 8px', borderRadius: 6, fontSize: 12 }}>.\venv\Scripts\activate</code></li>
              <li>Şu komutu çalıştırın: <code style={{ background: colors.preBg, color: colors.preText, padding: '2px 8px', borderRadius: 6, fontSize: 12 }}>python realtime_nids_scapy.py</code></li>
            </ol>
            <p style={{ margin: '14px 0 0 0', fontSize: 12, color: colors.textMuted }}>
              Normal trafikte az uyarı, gerçekten sıra dışı akışlarda daha fazla uyarı görürsünüz.
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
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ color: theme === 'light' ? '#6366f1' : '#c4b5fd', fontSize: 13 }}>Yüklü: {apiModels.models.length ? apiModels.models.map((m) => modelLabels[m] || m).join(', ') : 'yok'}</span>
              {apiModels.ensemble_available && <span style={{ padding: '4px 10px', borderRadius: 999, background: theme === 'light' ? 'rgba(99,102,241,0.15)' : 'rgba(167,139,250,0.2)', border: theme === 'light' ? '1px solid rgba(99,102,241,0.4)' : '1px solid rgba(167,139,250,0.5)', fontSize: 12, color: colors.text }}>Ensemble kullanılabilir</span>}
            </div>
          )}
          {comparison?.available && comparison.rows?.length > 0 ? (
            <div style={{ overflowX: 'auto', borderRadius: 14, border: `1px solid ${colors.inputBorder}`, background: colors.boxBg, marginBottom: 14 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Model</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Accuracy</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>F1</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Recall</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Precision</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>ROC-AUC</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>FPR</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row: Record<string, unknown>, i: number) => (
                    <tr key={i}>
                      <td style={{ padding: '10px 14px', color: colors.text }}>{modelLabels[String(row.model)] || String(row.model)}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: colors.textMuted }}>{typeof row.accuracy === 'number' ? (row.accuracy as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: colors.textMuted }}>{typeof row.f1 === 'number' ? (row.f1 as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: colors.textMuted }}>{typeof row.recall === 'number' ? (row.recall as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: colors.textMuted }}>{typeof row.precision === 'number' ? (row.precision as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: colors.textMuted }}>{typeof row.roc_auc === 'number' ? (row.roc_auc as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: colors.textMuted }}>{typeof row.false_positive_rate === 'number' ? (row.false_positive_rate as number).toFixed(4) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ background: colors.boxBg, padding: 14, borderRadius: 12, border: `1px solid ${colors.inputBorder}`, fontSize: 13, color: colors.textMuted, marginBottom: 14 }}>
              Karşılaştırma verisi yok. Lokalde proje kökünde: <code style={{ display: 'block', marginTop: 6, background: colors.preBg, color: colors.preText, padding: '8px 10px', borderRadius: 8 }}>python scripts\experiements\compare_core_models.py</code>
              Sonra backend&#39;i yeniden başlatıp sayfayı yenileyin.
            </div>
          )}
          <div style={{ marginBottom: 14, padding: 14, borderRadius: 12, background: theme === 'light' ? 'rgba(99,102,241,0.08)' : 'rgba(59,130,246,0.08)', border: theme === 'light' ? '1px solid rgba(99,102,241,0.25)' : '1px solid rgba(59,130,246,0.25)', fontSize: 12, color: colors.text, lineHeight: 1.55 }}>
            <strong style={{ color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Metrikleri nasıl okursunuz?</strong> F1 ve Accuracy yüksek (örn. &gt;0,7), FPR düşükse model güvenilir sayılır. ROC-AUC 1&#39;e yakınsa anomali ile normal trafiği iyi ayırıyor demektir.
          </div>
          <div style={{ overflowX: 'auto', borderRadius: 12, border: `1px solid ${colors.inputBorder}`, background: colors.boxBg }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '10px 12px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Model</th>
                  <th style={{ textAlign: 'left', padding: '10px 12px', color: theme === 'light' ? '#4f46e5' : '#93c5fd' }}>Sunucu durumu</th>
                </tr>
              </thead>
              <tbody>
                {['standard_ae', 'isolation_forest', 'one_class_svm', 'lstm_ae', 'ensemble'].map((key) => (
                  <tr key={key}>
                    <td style={{ padding: '10px 12px', color: colors.text }}>{modelLabels[key] || key}</td>
                    <td style={{ padding: '10px 12px' }}>
                      {key === 'ensemble' ? (
                        apiModels?.ensemble_available ? <span style={{ color: '#059669' }}>Kullanılabilir</span> : <span style={{ color: colors.textMuted }}>En az bir model gerekli</span>
                      ) : (
                        apiModels?.models?.includes(key) ? <span style={{ color: '#059669' }}>Yüklü</span> : <span style={{ color: colors.textMuted }}>Yok</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  )
}
