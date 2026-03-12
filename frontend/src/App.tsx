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
  const [monitoring, setMonitoring] = useState(false)
  const [healthStatus, setHealthStatus] = useState<'up' | 'down' | null>(null)
  const [lastCheck, setLastCheck] = useState<string | null>(null)
  const [isChecking, setIsChecking] = useState(false)
  const [isScoring, setIsScoring] = useState(false)
  const [scoreNote, setScoreNote] = useState<string>('') // skorun yorumu

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
          ]
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
    // İlk açılışta bir kere health kontrol et
    checkHealth()
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
    <div style={{ padding: 24, maxWidth: 600 }}>
      <h1>🔒 Network Anomaly Detection (PWA)</h1>
      <p>Backend bağlantısını test et ve canlı durumunu izle.</p>
      {!API_BASE && !isLocal && (
        <p style={{ color: '#fbbf24', fontSize: 14 }}>
          ⚠️ Canlıda backend URL ayarlı değil. Vercel → Project → Settings → Environment Variables → <code>VITE_API_URL</code> = Render backend adresi (örn. https://b2-xxx.onrender.com), sonra Redeploy.
        </p>
      )}
      <div style={{ display: 'flex', gap: 12, marginBottom: 16, alignItems: 'center' }}>
        <button onClick={checkHealth} disabled={isChecking}>
          {isChecking ? 'Health kontrol (bekle)...' : 'Health kontrol'}
        </button>
        <button onClick={testScore} disabled={isScoring}>
          {isScoring ? 'Skor test (bekle)...' : 'Skor test'}
        </button>
        <button
          onClick={() => setMonitoring((m) => !m)}
          style={{ background: monitoring ? '#10b981' : undefined }}
        >
          {monitoring ? 'Canlı izlemeyi durdur' : 'Canlı health izle'}
        </button>
      </div>
      <div style={{ marginBottom: 16, fontSize: 14, display: 'flex', gap: 12, alignItems: 'center' }}>
        <span>
          Durum:{' '}
          {healthStatus === 'up' && <span style={{ color: '#22c55e' }}>UP ✅</span>}
          {healthStatus === 'down' && <span style={{ color: '#ef4444' }}>DOWN ❌</span>}
          {healthStatus === null && <span style={{ color: '#9ca3af' }}>bilinmiyor</span>}
        </span>
        {lastCheck && <span>Son kontrol: {lastCheck}</span>}
      </div>
      {health && <pre style={{ background: '#1e293b', padding: 12, borderRadius: 8 }}>{health}</pre>}
      {scores !== null && (
        <div style={{ marginTop: 12 }}>
          <p>Skorlar (örnek trafik): {scores.join(', ')}</p>
          {scoreNote && <p style={{ fontSize: 14, color: '#e5e7eb' }}>{scoreNote}</p>}
        </div>
      )}
    </div>
  )
}
