/**
 * api.ts — Typed client for all non-streaming API calls.
 * Streaming (SSE) is handled separately in hooks/useChat.ts.
 */

import type { ApiMessage, Session } from '../types'

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
