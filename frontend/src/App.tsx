import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || ''

export default function App() {
  const [health, setHealth] = useState<string>('')
  const [scores, setScores] = useState<number[] | null>(null)

  const checkHealth = async () => {
    try {
      const r = await fetch(`${API_BASE}/health`)
      const j = await r.json()
      setHealth(JSON.stringify(j, null, 2))
    } catch (e) {
      setHealth('Hata: ' + String(e))
    }
  }

  const testScore = async () => {
    try {
      const r = await fetch(`${API_BASE}/api/score`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: [
            [0, 100, 200, 5, 10, 300, 15, 0.5, 0.5, 0.1, 0.2, 2]
          ]
        })
      })
      const j = await r.json()
      setScores(j.scores || null)
    } catch (e) {
      setScores(null)
      setHealth('Score hatası: ' + String(e))
    }
  }

  return (
    <div style={{ padding: 24, maxWidth: 600 }}>
      <h1>🔒 Network Anomaly Detection (PWA)</h1>
      <p>Backend bağlantısını test et.</p>
      <div style={{ display: 'flex', gap: 12, marginBottom: 16 }}>
        <button onClick={checkHealth}>Health kontrol</button>
        <button onClick={testScore}>Skor test</button>
      </div>
      {health && <pre style={{ background: '#1e293b', padding: 12, borderRadius: 8 }}>{health}</pre>}
      {scores !== null && <p>Skorlar: {scores.join(', ')}</p>}
    </div>
  )
}
