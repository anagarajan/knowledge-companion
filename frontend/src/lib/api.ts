/**
 * api.ts — Typed client for all non-streaming API calls.
 * Streaming (SSE) is handled separately in hooks/useChat.ts.
 */

import type { ApiMessage, Session, GraphEntity, GraphRelationship, GraphStats, PatientListResponse, PatientDetail, PatientStats } from '../types'

const BASE = '/api'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`API ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

// ── Sessions ──────────────────────────────────────────────────────────────────

export function listSessions(): Promise<Session[]> {
  return request<Session[]>('/sessions')
}

export function createSession(folders: string[] = [], title = 'New Chat'): Promise<Session> {
  return request<Session>('/sessions', {
    method: 'POST',
    body: JSON.stringify({ folders, title }),
  })
}

export function deleteSession(id: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/sessions/${id}`, { method: 'DELETE' })
}

export function clearSession(id: string): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/sessions/${id}/clear`, { method: 'POST' })
}

export function updateFolders(id: string, folders: string[]): Promise<{ ok: boolean }> {
  return request<{ ok: boolean }>(`/sessions/${id}/folders`, {
    method: 'PATCH',
    body: JSON.stringify({ folders }),
  })
}

// ── Messages ──────────────────────────────────────────────────────────────────

export function getMessages(sessionId: string): Promise<ApiMessage[]> {
  return request<ApiMessage[]>(`/sessions/${sessionId}/messages`)
}

// ── Folders ───────────────────────────────────────────────────────────────────

export function listFolders(): Promise<string[]> {
  return request<string[]>('/folders')
}

// ── Knowledge Graph ──────────────────────────────────────────────────────────

export function listEntities(params: {
  search?: string
  type?: string
  folder?: string
  limit?: number
  offset?: number
} = {}): Promise<GraphEntity[]> {
  const query = new URLSearchParams()
  if (params.search) query.set('search', params.search)
  if (params.type)   query.set('type', params.type)
  if (params.folder) query.set('folder', params.folder)
  if (params.limit)  query.set('limit', String(params.limit))
  if (params.offset) query.set('offset', String(params.offset))
  const qs = query.toString()
  return request<GraphEntity[]>(`/graph/entities${qs ? `?${qs}` : ''}`)
}

export function getEntity(id: string): Promise<{
  entity: GraphEntity
  relationships: GraphRelationship[]
}> {
  return request(`/graph/entities/${id}`)
}

export function getRelatedEntities(
  id: string,
  depth = 2,
): Promise<GraphEntity[]> {
  return request<GraphEntity[]>(`/graph/entities/${id}/related?depth=${depth}`)
}

export function getGraphStats(): Promise<GraphStats> {
  return request<GraphStats>('/graph/stats')
}

// ── Patients (Track 3) ────────────────────────────────────────────────────────

export function listPatients(params: {
  name?: string
  gender?: string
  city?: string
  state?: string
  medication?: string
  icd10?: string
  limit?: number
  offset?: number
} = {}): Promise<PatientListResponse> {
  const query = new URLSearchParams()
  if (params.name)       query.set('name', params.name)
  if (params.gender)     query.set('gender', params.gender)
  if (params.city)       query.set('city', params.city)
  if (params.state)      query.set('state', params.state)
  if (params.medication) query.set('medication', params.medication)
  if (params.icd10)      query.set('icd10', params.icd10)
  if (params.limit)      query.set('limit', String(params.limit))
  if (params.offset)     query.set('offset', String(params.offset))
  const qs = query.toString()
  return request<PatientListResponse>(`/patients${qs ? `?${qs}` : ''}`)
}

export function getPatient(patientId: string): Promise<PatientDetail> {
  return request<PatientDetail>(`/patients/${patientId}`)
}

export function getPatientStats(): Promise<PatientStats> {
  return request<PatientStats>('/patients/stats')
}
