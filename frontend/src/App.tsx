import { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''
const isLocal =
  typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1')

function buildUrl(path: string) {
  // Prod'da tam URL, local'de proxy ile göreli path kullan
  if (API_BASE) return `${API_BASE}${path}`
  return path
}

export default function App() {
  const [health, setHealth] = useState<string>('')
  const [scores, setScores] = useState<number[] | null>(null)
  const [selectedModel, setSelectedModel] = useState<'auto' | 'ensemble' | 'standard_ae' | 'isolation_forest'>('auto')
  const [monitoring, setMonitoring] = useState(false)
  const [healthStatus, setHealthStatus] = useState<'up' | 'down' | null>(null)
  const [lastCheck, setLastCheck] = useState<string | null>(null)
  const [isChecking, setIsChecking] = useState(false)
  const [isScoring, setIsScoring] = useState(false)
  const [scoreNote, setScoreNote] = useState<string>('') // skorun yorumu
  const [apiModels, setApiModels] = useState<{ models: string[]; ensemble_available: boolean } | null>(null)

  const fetchModels = async () => {
    try {
      const r = await fetch(buildUrl('/api/models'))
      if (r.ok) {
        const j = await r.json()
        setApiModels(j)
      }
    } catch {
      setApiModels(null)
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
      const r = await fetch(buildUrl('/api/score'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: [
            [0, 100, 200, 5, 10, 300, 15, 0.5, 0.5, 0.1, 0.2, 2]
          ],
          model: selectedModel === 'auto' ? null : selectedModel
        })
      })
      const text = await r.text()
      if (!r.ok) {
        setHealth(`Score hatası ${r.status}: ${text.slice(0, 200)}`)
        setScores(null)
        return
      }
      const j = JSON.parse(text)
      setScores(j.scores || null)
      // Bu skorlar demo / örnek trafik içindir, kullanıcının makinesi değildir
      if (j.model === 'demo') {
        setScoreNote(
          'Bu skorlar demo / örnek trafik içindir, bilgisayarına ait gerçek trafik değildir.'
        )
      } else {
        const s = Array.isArray(j.scores) && j.scores.length > 0 ? j.scores[0] : null
        if (s == null) {
          setScoreNote('')
        } else if (s < 0.3) {
          setScoreNote('Skor düşük: Örnek trafik normal aralıkta görünüyor.')
        } else if (s < 0.7) {
          setScoreNote('Skor orta: Örnek trafik şüpheli olabilir.')
        } else {
          setScoreNote('Skor yüksek: Örnek trafik anomali olarak değerlendirilebilir.')
        }
      }
    } catch (e) {
      setScores(null)
      setHealth('Score hatası: ' + String(e))
    } finally {
      setIsScoring(false)
    }
  }

  useEffect(() => {
    checkHealth()
    fetchModels()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!monitoring) return
    const id = window.setInterval(() => {
      checkHealth()
    }, 10000) // 10 sn'de bir health kontrol
    return () => window.clearInterval(id)
  }, [monitoring])

  return (
    <div
      style={{
        minHeight: '100vh',
        background:
          'radial-gradient(circle at top left, #fce7f3 0, #0f172a 40%, #0b1120 100%)',
        color: '#e5e7eb',
        padding: 24
      }}
    >
      <div
        style={{
          maxWidth: 980,
          margin: '0 auto',
          backdropFilter: 'blur(12px)',
          borderRadius: 24,
          border: '1px solid rgba(148, 163, 184, 0.25)',
          background:
            'linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.9), rgba(244,114,182,0.06))',
          boxShadow: '0 24px 60px rgba(15,23,42,0.8)',
          padding: 24
        }}
      >
        <header
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginBottom: 20,
            gap: 12,
            flexWrap: 'wrap'
          }}
        >
          <div>
            <h1
              style={{
                fontSize: 28,
                fontWeight: 700,
                margin: 0,
                background:
                  'linear-gradient(120deg, #f9a8d4, #93c5fd)',
                WebkitBackgroundClip: 'text',
                color: 'transparent'
              }}
            >
              Network Anomaly Detection
            </h1>
            <p style={{ color: '#9ca3af', marginTop: 6 }}>
              Backend durumunu izle, farklı modelleri dene ve skorları karşılaştır.
            </p>
          </div>
          <div
            style={{
              padding: '6px 12px',
              borderRadius: 999,
              background:
                healthStatus === 'up'
                  ? 'rgba(16,185,129,0.12)'
                  : healthStatus === 'down'
                  ? 'rgba(248,113,113,0.12)'
                  : 'rgba(148,163,184,0.12)',
              border:
                healthStatus === 'up'
                  ? '1px solid rgba(52,211,153,0.6)'
                  : healthStatus === 'down'
                  ? '1px solid rgba(248,113,113,0.6)'
                  : '1px solid rgba(148,163,184,0.5)',
              fontSize: 13,
              display: 'flex',
              alignItems: 'center',
              gap: 8
            }}
          >
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: '50%',
                background:
                  healthStatus === 'up'
                    ? '#22c55e'
                    : healthStatus === 'down'
                    ? '#ef4444'
                    : '#9ca3af',
                boxShadow:
                  healthStatus === 'up'
                    ? '0 0 10px rgba(34,197,94,0.8)'
                    : 'none'
              }}
            />
            <span>
              API durumu:{' '}
              {healthStatus === 'up' && 'UP'}
              {healthStatus === 'down' && 'DOWN'}
              {healthStatus === null && 'bilinmiyor'}
            </span>
            {lastCheck && <span style={{ opacity: 0.7 }}>• {lastCheck}</span>}
          </div>
        </header>
      {!API_BASE && !isLocal && (
        <p style={{ color: '#fbbf24', fontSize: 14 }}>
          ⚠️ Canlıda backend URL ayarlı değil. Vercel → Project → Settings → Environment Variables → <code>VITE_API_URL</code> = Render backend adresi (örn. https://b2-xxx.onrender.com), sonra Redeploy.
        </p>
      )}
      <div
        style={{
          display: 'flex',
          gap: 12,
          marginBottom: 16,
          alignItems: 'center',
          flexWrap: 'wrap'
        }}
      >
        <button
          onClick={checkHealth}
          disabled={isChecking}
          style={{
            padding: '8px 14px',
            borderRadius: 999,
            border: 'none',
            background: '#0ea5e9',
            color: 'white',
            cursor: isChecking ? 'default' : 'pointer'
          }}
        >
          {isChecking ? 'Health kontrol (bekle)...' : 'Health kontrol'}
        </button>
        <button
          onClick={testScore}
          disabled={isScoring}
          style={{
            padding: '8px 14px',
            borderRadius: 999,
            border: 'none',
            background: '#22c55e',
            color: 'white',
            cursor: isScoring ? 'default' : 'pointer'
          }}
        >
          {isScoring ? 'Skor test (bekle)...' : 'Skor test'}
        </button>
        <button
          onClick={() => setMonitoring((m) => !m)}
          style={{
            padding: '8px 14px',
            borderRadius: 999,
            border: '1px solid #4b5563',
            background: monitoring ? '#10b981' : 'transparent',
            color: monitoring ? 'white' : '#e5e7eb',
            cursor: 'pointer'
          }}
        >
          {monitoring ? 'Canlı izlemeyi durdur' : 'Canlı health izle'}
        </button>
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 14, color: '#9ca3af' }}>Model:</span>
          <select
            value={selectedModel}
            onChange={(e) =>
              setSelectedModel(e.target.value as 'auto' | 'ensemble' | 'standard_ae' | 'isolation_forest')
            }
            style={{
              background: '#020617',
              color: '#e5e7eb',
              borderRadius: 999,
              border: '1px solid #4b5563',
              padding: '6px 10px',
              fontSize: 14
            }}
          >
            <option value="auto">Otomatik (ensemble → AE → IF)</option>
            <option value="ensemble">Ensemble</option>
            <option value="standard_ae">Standard Autoencoder</option>
            <option value="isolation_forest">Isolation Forest</option>
          </select>
        </div>
      </div>
      {health && (
        <pre
          style={{
            background:
              'linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,23,42,0.95))',
            padding: 12,
            borderRadius: 8,
            border: '1px solid rgba(148,163,184,0.4)',
            fontSize: 13,
            maxHeight: 260,
            overflow: 'auto'
          }}
        >
          {health}
        </pre>
      )}
      {scores !== null && (
        <div
          style={{
            marginTop: 16,
            padding: 14,
            borderRadius: 16,
            background:
              'linear-gradient(135deg, rgba(248,113,165,0.18), rgba(59,130,246,0.12))',
            border: '1px solid rgba(248,113,165,0.45)'
          }}
        >
          <p>Skorlar (örnek trafik): {scores.join(', ')}</p>
          {scoreNote && (
            <p style={{ fontSize: 14, color: '#fee2e2', marginTop: 4 }}>{scoreNote}</p>
          )}
        </div>
      )}

        <section
          style={{
            marginTop: 24,
            padding: 20,
            borderRadius: 20,
            background:
              'linear-gradient(145deg, rgba(244,114,182,0.12), rgba(96,165,250,0.1))',
            border: '1px solid rgba(244,114,182,0.35)',
            boxShadow: '0 8px 32px rgba(15,23,42,0.4)'
          }}
        >
          <h2
            style={{
              margin: '0 0 12px 0',
              fontSize: 18,
              fontWeight: 600,
              background: 'linear-gradient(90deg, #f9a8d4, #93c5fd)',
              WebkitBackgroundClip: 'text',
              color: 'transparent'
            }}
          >
            Model karşılaştırma
          </h2>
          <p style={{ color: '#94a3b8', fontSize: 13, marginBottom: 12 }}>
            Sunucuda yüklü modeller ve ensemble durumu. Tam metrik karşılaştırması için proje kökünde:{' '}
            <code style={{ background: 'rgba(15,23,42,0.6)', padding: '2px 6px', borderRadius: 6 }}>python scripts\experiements\compare_core_models.py</code>
          </p>
          {apiModels && (
            <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ color: '#c4b5fd', fontSize: 14 }}>
                Yüklü: {apiModels.models.length ? apiModels.models.join(', ') : '—'}
              </span>
              {apiModels.ensemble_available && (
                <span
                  style={{
                    padding: '4px 10px',
                    borderRadius: 999,
                    background: 'rgba(167,139,250,0.25)',
                    border: '1px solid rgba(167,139,250,0.5)',
                    fontSize: 13
                  }}
                >
                  Ensemble kullanılabilir
                </span>
              )}
            </div>
          )}
          <div
            style={{
              marginTop: 12,
              overflowX: 'auto',
              borderRadius: 12,
              border: '1px solid rgba(148,163,184,0.3)',
              background: 'rgba(15,23,42,0.5)'
            }}
          >
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '10px 12px', color: '#93c5fd' }}>Model</th>
                  <th style={{ textAlign: 'left', padding: '10px 12px', color: '#93c5fd' }}>Durum</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>Standard Autoencoder</td>
                  <td style={{ padding: '10px 12px' }}>
                    {apiModels?.models?.includes('standard_ae') ? (
                      <span style={{ color: '#86efac' }}>Yüklü</span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>Dosya yok</span>
                    )}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>Isolation Forest</td>
                  <td style={{ padding: '10px 12px' }}>
                    {apiModels?.models?.includes('isolation_forest') ? (
                      <span style={{ color: '#86efac' }}>Yüklü</span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>Dosya yok</span>
                    )}
                  </td>
                </tr>
                <tr>
                  <td style={{ padding: '10px 12px', color: '#e2e8f0' }}>Ensemble</td>
                  <td style={{ padding: '10px 12px' }}>
                    {apiModels?.ensemble_available ? (
                      <span style={{ color: '#86efac' }}>Kullanılabilir</span>
                    ) : (
                      <span style={{ color: '#94a3b8' }}>En az bir model gerekli</span>
                    )}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  )
}
