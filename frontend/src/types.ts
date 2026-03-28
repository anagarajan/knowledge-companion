// ── API types (mirror the FastAPI response shapes) ────────────────────────────

export interface Session {
  id:         string
  title:      string
  folders:    string[]
  created_at: string
  updated_at: string
  is_active:  boolean
}

export interface Source {
  filename: string
  page:     number
  score:    number
  was_ocr:  boolean
  folder:   string
}

export interface Confidence {
  level:  'HIGH' | 'MEDIUM' | 'LOW'
  reason: string
}

export interface ApiMessage {
  id:         string
  session_id: string
  role:       'user' | 'assistant'
  content:    string
  sources:    Source[]
  confidence: Confidence | null
  model_used: string
  timestamp:  string
}

// ── Local UI types ─────────────────────────────────────────────────────────────

/** Message as held in React state — before and after it's persisted. */
export interface Message {
  id:          string
  role:        'user' | 'assistant'
  content:     string
  sources:     Source[]
  confidence:  Confidence | null
  model_used:  string
  isStreaming?: boolean   // true while tokens are still arriving
}
