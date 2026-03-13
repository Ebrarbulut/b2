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

const card = {
  marginTop: 24,
  padding: '24px 22px',
  borderRadius: 18,
  background: 'linear-gradient(145deg, rgba(244,114,182,0.07), rgba(96,165,250,0.05))',
  border: '1px solid rgba(244,114,182,0.22)',
  boxShadow: '0 6px 28px rgba(15,23,42,0.2)',
} as const

const title = {
  margin: '0 0 6px 0',
  fontSize: 17,
  fontWeight: 600,
  background: 'linear-gradient(90deg, #f9a8d4, #93c5fd)',
  WebkitBackgroundClip: 'text' as const,
  color: 'transparent',
}

const desc = { color: '#94a3b8', fontSize: 13, marginBottom: 14 } as const

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
  const [backgroundMode, setBackgroundMode] = useState<'off' | 'interval' | 'hours'>('off')
  const [hourStart, setHourStart] = useState(9)
  const [hourEnd, setHourEnd] = useState(18)
  const [showHelp, setShowHelp] = useState(false)
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

  useEffect(() => {
    checkHealth()
    fetchModels()
    fetchComparison()
  }, [])

  useEffect(() => {
    if (backgroundMode === 'off') return
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

  return (
    <div
      style={{
        minHeight: '100vh',
        background: 'radial-gradient(ellipse 80% 50% at 50% -20%, rgba(248,113,182,0.2), transparent), radial-gradient(ellipse 60% 40% at 100% 50%, rgba(96,165,250,0.12), transparent), #0f172a',
        color: '#e5e7eb',
        padding: '20px 16px 40px',
      }}
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
                background: 'linear-gradient(120deg, #f9a8d4, #93c5fd)',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
              }}
            >
              Ağ Anomali Tespiti
            </h1>
            <p style={{ color: '#94a3b8', marginTop: 6, fontSize: 14, maxWidth: 420 }}>
              Trafiğinizin normal mi yoksa şüpheli mi olduğunu yapay zeka modelleriyle kontrol edin. İsterseniz kendi bilgisayarınızda canlı analiz de yapabilirsiniz.
            </p>
            <p style={{ color: '#64748b', marginTop: 6, fontSize: 12, maxWidth: 420 }}>
              Bu uygulama PWA olarak indirilebilir: bilgisayar veya telefonda tarayıcıdan &quot;Uygulamayı yükle&quot; ile kurup uygulama gibi kullanabilirsiniz (masaüstü .exe değildir).
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
            <div
              style={{
                padding: '8px 14px',
                borderRadius: 12,
                background: healthStatus === 'up' ? 'rgba(34,197,94,0.15)' : healthStatus === 'down' ? 'rgba(239,68,68,0.15)' : 'rgba(148,163,184,0.15)',
                border: `1px solid ${healthStatus === 'up' ? 'rgba(34,197,94,0.5)' : healthStatus === 'down' ? 'rgba(239,68,68,0.5)' : 'rgba(148,163,184,0.4)'}`,
                fontSize: 13,
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}
            >
              <span style={{ width: 8, height: 8, borderRadius: '50%', background: healthStatus === 'up' ? '#22c55e' : healthStatus === 'down' ? '#ef4444' : '#94a3b8' }} />
              Sunucu: {healthStatus === 'up' ? 'Bağlı' : healthStatus === 'down' ? 'Bağlı değil' : '—'} {lastCheck && ` · ${lastCheck}`}
            </div>
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
          <p style={{ color: '#fbbf24', fontSize: 13, marginBottom: 16 }}>
            Canlıda backend adresi ayarlı değil. Vercel → Environment Variables → VITE_API_URL ekleyip yeniden deploy edin.
          </p>
        )}

        {/* 1. Bağlantı ve örnek analiz */}
        <section style={card}>
          <h2 style={title}>Bağlantı ve örnek analiz</h2>
          <p style={desc}>
            Sunucuya ulaşılabiliyor mu görmek için bağlantıyı test edin. Örnek analiz ile seçtiğiniz modelin bir trafik örneğine verdiği risk skorunu görebilirsiniz.
          </p>
          <button type="button" onClick={() => setShowHelp((h) => !h)} style={{ marginBottom: 12, padding: '6px 12px', fontSize: 12, background: 'rgba(148,163,184,0.2)', border: '1px solid rgba(148,163,184,0.4)', borderRadius: 8, color: '#93c5fd', cursor: 'pointer' }}>
            {showHelp ? 'Ne işe yarar? (kapat)' : 'Ne işe yarar?'}
          </button>
          {showHelp && (
            <div style={{ marginBottom: 14, padding: 14, background: 'rgba(15,23,42,0.6)', borderRadius: 12, border: '1px solid rgba(148,163,184,0.2)', fontSize: 13, color: '#cbd5e1' }}>
              <p style={{ margin: '0 0 8px 0' }}><strong>Bağlantıyı kontrol et:</strong> Uygulama sunucuya (API) bir istek gönderir. Sunucu &quot;çalışıyorum&quot; diye yanıt verirse bağlantı sağlam demektir; böylece sistemin canlı olup olmadığını anlarsınız.</p>
              <p style={{ margin: 0 }}><strong>Örnek analiz dene:</strong> Uygulama sunucuya sabit bir örnek trafik vektörü (sayılar listesi) gönderir. Sunucudaki model bu veriyi analiz edip bir risk skoru üretir. Bu, gerçek trafiğiniz değildir; sadece modelin çalıştığını ve örnek veriye nasıl yanıt verdiğini görmeniz içindir.</p>
            </div>
          )}
          <div style={{ display: 'flex', gap: 10, marginBottom: 10, flexWrap: 'wrap', alignItems: 'center' }}>
            <div>
              <button onClick={checkHealth} disabled={isChecking} style={{ padding: '10px 18px', borderRadius: 12, border: 'none', background: '#0ea5e9', color: 'white', cursor: isChecking ? 'default' : 'pointer', fontWeight: 500 }}>
                {isChecking ? 'Kontrol ediliyor...' : 'Bağlantıyı kontrol et'}
              </button>
              <p style={{ margin: '4px 0 0 0', fontSize: 11, color: '#64748b', maxWidth: 220 }}>Sunucuya bir istek atar; yanıt &quot;ok&quot; ise API ayakta demektir.</p>
            </div>
            <div>
              <button onClick={testScore} disabled={isScoring} style={{ padding: '10px 18px', borderRadius: 12, border: 'none', background: '#22c55e', color: 'white', cursor: isScoring ? 'default' : 'pointer', fontWeight: 500 }}>
                {isScoring ? 'Analiz ediliyor...' : 'Örnek analiz dene'}
              </button>
              <p style={{ margin: '4px 0 0 0', fontSize: 11, color: '#64748b', maxWidth: 220 }}>Sabit örnek veriyi modele gönderir; dönen risk skoru modelin canlı olduğunu gösterir.</p>
            </div>
            <div style={{ marginLeft: 4 }}>
              <span style={{ fontSize: 12, color: '#94a3b8', marginRight: 8 }}>Arka plan kontrolü:</span>
              <select
                value={backgroundMode}
                onChange={(e) => setBackgroundMode(e.target.value as 'off' | 'interval' | 'hours')}
                style={{ background: 'rgba(15,23,42,0.7)', color: '#e5e7eb', border: '1px solid rgba(148,163,184,0.4)', borderRadius: 8, padding: '6px 10px', fontSize: 12 }}
              >
                <option value="off">Kapalı</option>
                <option value="interval">Her X dakikada bir</option>
                <option value="hours">Sadece belirli saatlerde</option>
              </select>
              {backgroundMode === 'interval' && (
                <span style={{ marginLeft: 8 }}>
                  <select
                    value={checkIntervalMinutes}
                    onChange={(e) => setCheckIntervalMinutes(Number(e.target.value))}
                    style={{ background: 'rgba(15,23,42,0.7)', color: '#e5e7eb', border: '1px solid rgba(148,163,184,0.4)', borderRadius: 8, padding: '4px 8px', fontSize: 12 }}
                  >
                    <option value={1}>1 dk</option>
                    <option value={5}>5 dk</option>
                    <option value={15}>15 dk</option>
                    <option value={30}>30 dk</option>
                  </select>
                </span>
              )}
              {backgroundMode === 'hours' && (
                <span style={{ marginLeft: 8, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                  <label style={{ fontSize: 12, color: '#94a3b8' }}>Saat</label>
                  <select value={hourStart} onChange={(e) => setHourStart(Number(e.target.value))} style={{ background: 'rgba(15,23,42,0.7)', color: '#e5e7eb', border: '1px solid rgba(148,163,184,0.4)', borderRadius: 8, padding: '4px 6px', fontSize: 12 }}>
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                    ))}
                  </select>
                  <span style={{ color: '#64748b' }}>–</span>
                  <select value={hourEnd} onChange={(e) => setHourEnd(Number(e.target.value))} style={{ background: 'rgba(15,23,42,0.7)', color: '#e5e7eb', border: '1px solid rgba(148,163,184,0.4)', borderRadius: 8, padding: '4px 6px', fontSize: 12 }}>
                    {Array.from({ length: 24 }, (_, i) => (
                      <option key={i} value={i}>{String(i).padStart(2, '0')}:00</option>
                    ))}
                  </select>
                </span>
              )}
            </div>
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 13, color: '#94a3b8' }}>Örnek analizde kullanılacak model:</span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                style={{ background: 'rgba(15,23,42,0.8)', color: '#e5e7eb', borderRadius: 12, border: '1px solid rgba(148,163,184,0.4)', padding: '8px 12px', fontSize: 13 }}
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
              <p style={{ margin: '0 0 6px 0', fontSize: 12, color: '#94a3b8' }}>Sunucu yanıtı (bağlantı sağlamsa &quot;ok&quot; görürsünüz)</p>
              <pre style={{ background: 'rgba(15,23,42,0.7)', padding: 12, borderRadius: 12, border: '1px solid rgba(148,163,184,0.25)', fontSize: 12, maxHeight: 160, overflow: 'auto', margin: 0 }}>
                {health}
              </pre>
            </div>
          )}
          {scores !== null && (
            <div style={{ marginTop: 16, padding: 18, borderRadius: 14, background: 'linear-gradient(135deg, rgba(248,113,165,0.1), rgba(59,130,246,0.06))', border: '1px solid rgba(248,113,165,0.3)' }}>
              <p style={{ margin: '0 0 6px 0', fontSize: 12, color: '#94a3b8' }}>Örnek analiz sonucu</p>
              <p style={{ margin: 0, fontSize: 14 }}>
                <strong>Risk skoru:</strong> {scores.map((x) => (typeof x === 'number' ? x.toFixed(4) : x)).join(', ')}
                {usedModelName && <span style={{ color: '#93c5fd', marginLeft: 8 }}>(kullanılan model: {modelLabels[usedModelName] || usedModelName})</span>}
              </p>
              {scoreNote && <p style={{ fontSize: 13, color: '#e2e8f0', marginTop: 8, marginBottom: 0 }}>{scoreNote}</p>}
            </div>
          )}
        </section>

        {/* 2. Kendi ağını analiz et */}
        <section style={card}>
          <h2 style={title}>Kendi bilgisayarınızda canlı analiz</h2>
          <p style={desc}>
            İsterseniz bu programı kendi bilgisayarınızda çalıştırıp ağ trafiğinizi arka planda analiz ettirebilirsiniz. Şüpheli görünen bağlantılar konsolda işaretlenir; bilgisayarınızı sürekli meşgul etmez, sadece paketler geldikçe değerlendirilir.
          </p>
          <div style={{ background: 'rgba(15,23,42,0.45)', padding: 18, borderRadius: 14, border: '1px solid rgba(148,163,184,0.18)', fontSize: 13 }}>
            <p style={{ margin: '0 0 10px 0', color: '#e2e8f0' }}><strong>Nasıl yapılır?</strong></p>
            <ol style={{ margin: 0, paddingLeft: 20, color: '#cbd5e1', lineHeight: 1.85 }}>
              <li>PowerShell veya terminali <strong>Yönetici olarak çalıştır</strong>.</li>
              <li>Proje klasörüne gidin; sanal ortamı açın: <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 6 }}>.\venv\Scripts\activate</code></li>
              <li>Şu komutu yazın: <code style={{ background: 'rgba(0,0,0,0.3)', padding: '2px 6px', borderRadius: 6 }}>python realtime_nids_scapy.py</code></li>
            </ol>
            <p style={{ margin: '14px 0 0 0', color: '#94a3b8', fontSize: 12 }}>
              Normal kullanımda az uyarı, gerçekten şüpheli trafikte daha fazla uyarı görmeniz beklenir. Durdurmak için Ctrl+C yeterli.
            </p>
          </div>
        </section>

        {/* 3. Model karşılaştırma */}
        <section style={card}>
          <h2 style={title}>Modeller ve performans</h2>
          <p style={desc}>
            Hangi modellerin yüklü olduğu ve (lokalde karşılaştırma script&#39;i çalıştırıldıysa) doğruluk, F1, hassasiyet gibi metrikler aşağıda.
          </p>
          {apiModels && (
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ color: '#c4b5fd', fontSize: 13 }}>Yüklü: {apiModels.models.length ? apiModels.models.map((m) => modelLabels[m] || m).join(', ') : 'yok'}</span>
              {apiModels.ensemble_available && <span style={{ padding: '4px 10px', borderRadius: 999, background: 'rgba(167,139,250,0.2)', border: '1px solid rgba(167,139,250,0.5)', fontSize: 12 }}>Ensemble kullanılabilir</span>}
            </div>
          )}
          {comparison?.available && comparison.rows?.length > 0 ? (
            <div style={{ overflowX: 'auto', borderRadius: 14, border: '1px solid rgba(148,163,184,0.25)', background: 'rgba(15,23,42,0.5)', marginBottom: 14 }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '12px 14px', color: '#93c5fd' }}>Model</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: '#93c5fd' }}>Accuracy</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: '#93c5fd' }}>F1</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: '#93c5fd' }}>Recall</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: '#93c5fd' }}>Precision</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: '#93c5fd' }}>ROC-AUC</th>
                    <th style={{ textAlign: 'right', padding: '12px 14px', color: '#93c5fd' }}>FPR</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.rows.map((row: Record<string, unknown>, i: number) => (
                    <tr key={i}>
                      <td style={{ padding: '10px 14px', color: '#e2e8f0' }}>{modelLabels[String(row.model)] || String(row.model)}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: '#cbd5e1' }}>{typeof row.accuracy === 'number' ? (row.accuracy as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: '#cbd5e1' }}>{typeof row.f1 === 'number' ? (row.f1 as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: '#cbd5e1' }}>{typeof row.recall === 'number' ? (row.recall as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: '#cbd5e1' }}>{typeof row.precision === 'number' ? (row.precision as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: '#cbd5e1' }}>{typeof row.roc_auc === 'number' ? (row.roc_auc as number).toFixed(4) : '—'}</td>
                      <td style={{ padding: '10px 14px', textAlign: 'right', color: '#cbd5e1' }}>{typeof row.false_positive_rate === 'number' ? (row.false_positive_rate as number).toFixed(4) : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div style={{ background: 'rgba(15,23,42,0.5)', padding: 14, borderRadius: 12, border: '1px solid rgba(148,163,184,0.2)', fontSize: 13, color: '#94a3b8', marginBottom: 14 }}>
              Karşılaştırma verisi yok. Lokalde proje kökünde: <code style={{ display: 'block', marginTop: 6, background: 'rgba(0,0,0,0.3)', padding: '8px 10px', borderRadius: 8 }}>python scripts\experiements\compare_core_models.py</code>
              Sonra backend&#39;i yeniden başlatıp sayfayı yenileyin.
            </div>
          )}
          <div style={{ marginBottom: 14, padding: 12, borderRadius: 12, background: 'rgba(59,130,246,0.08)', border: '1px solid rgba(59,130,246,0.25)', fontSize: 12, color: '#cbd5e1' }}>
            <strong style={{ color: '#93c5fd' }}>Model durumları nasıl yorumlanır?</strong> F1 ve Accuracy yüksek (örn. &gt;0,7), FPR (yanlış pozitif oranı) düşükse model iyi sayılır. ROC-AUC 1&#39;e yakınsa ayırt etme gücü yüksektir. Tabloda bu metrikleri karşılaştırarak hangi modelin sizin kullanımınıza daha uygun olduğunu görebilirsiniz.
          </div>
          <div style={{ overflowX: 'auto', borderRadius: 12, border: '1px solid rgba(148,163,184,0.25)', background: 'rgba(15,23,42,0.4)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '10px 12px', color: '#93c5fd' }}>Model</th>
                  <th style={{ textAlign: 'left', padding: '10px 12px', color: '#93c5fd' }}>Sunucu durumu</th>
                </tr>
              </thead>
              <tbody>
                {['standard_ae', 'isolation_forest', 'one_class_svm', 'lstm_ae', 'ensemble'].map((key) => (
                  <tr key={key}>
                    <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>{modelLabels[key] || key}</td>
                    <td style={{ padding: '10px 12px' }}>
                      {key === 'ensemble' ? (
                        apiModels?.ensemble_available ? <span style={{ color: '#86efac' }}>Kullanılabilir</span> : <span style={{ color: '#94a3b8' }}>En az bir model gerekli</span>
                      ) : (
                        apiModels?.models?.includes(key) ? <span style={{ color: '#86efac' }}>Yüklü</span> : <span style={{ color: '#94a3b8' }}>Yok</span>
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
