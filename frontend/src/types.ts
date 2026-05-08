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

// ── Knowledge Graph types ──────────────────────────────────────────────────────

export interface GraphEntity {
  id:              string
  name:            string
  name_normalized: string
  entity_type:     string
  source:          string
  folder:          string
  page:            number
  description:     string
  properties:      Record<string, unknown>
}

export interface GraphRelationship {
  id:               string
  source_entity_id: string
  target_entity_id: string
  relation_type:    string
  description:      string
  confidence:       number
  source_name?:     string   // resolved name (from GET /entities/{id})
  source_type?:     string
  target_name?:     string
  target_type?:     string
}

export interface GraphStats {
  total_entities:               number
  total_relationships:          number
  total_documents_with_metadata: number
  entities_by_type:             Record<string, number>
  relationships_by_type:        Record<string, number>
}

// ── Patient types (Track 3) ────────────────────────────────────────────────────

export interface PatientSummary {
  patient_id:         string
  full_name:          string | null
  date_of_birth:      string | null
  gender:             string | null
  city:               string | null
  state:              string | null
  insurance_provider: string | null
  insurance_id:       string | null
  icd10_codes:        string[]
  diagnoses:          string[]
  medications:        string[]
  last_extracted_at:  string | null
}

export interface PatientProvenance {
  field_name:   string
  field_value:  string
  source_file:  string
  source_page:  number
  confidence:   number
}

export interface PatientDetail extends PatientSummary {
  medical_history: string | null
  provenance:      PatientProvenance[]
}

export interface PatientListResponse {
  patients: PatientSummary[]
  total:    number
}

export interface PatientStats {
  total_patients:    number
  field_completeness: Record<string, number>
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
