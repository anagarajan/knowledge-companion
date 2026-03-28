/**
 * Sidebar.tsx — Left navigation panel.
 *
 * Three sections:
 *   1. Header + New Chat button
 *   2. Folder scope picker (checkboxes — which folders this session searches)
 *   3. Conversation history list
 */

import { Trash2, Plus, FolderOpen, MessageSquare } from 'lucide-react'
import { clsx } from 'clsx'
import type { Session } from '../types'

interface Props {
  sessions:         Session[]
  activeId:         string | null
  availableFolders: string[]
  selectedFolders:  string[]
  onNewChat:        () => void
  onSelectSession:  (id: string) => void
  onDeleteSession:  (id: string) => void
  onFolderToggle:   (folder: string, checked: boolean) => void
}

export default function Sidebar({
  sessions,
  activeId,
  availableFolders,
  selectedFolders,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onFolderToggle,
}: Props) {
  return (
    <aside className="flex flex-col w-64 h-full bg-sidebar border-r border-border shrink-0">

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="px-4 pt-5 pb-3 border-b border-border">
        <p className="text-sm font-semibold tracking-widest uppercase text-muted select-none">
          Knowledge Companion
        </p>
      </div>

      {/* ── New Chat button ─────────────────────────────────────────────────── */}
      <div className="px-3 pt-3">
        <button
          onClick={onNewChat}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg border border-border
                     text-sm text-black hover:bg-raised transition-colors"
        >
          <Plus size={15} />
          New chat
        </button>
      </div>

      {/* ── Folder picker ───────────────────────────────────────────────────── */}
      {availableFolders.length > 0 && (
        <div className="px-4 pt-4 pb-2">
          <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-widest text-muted mb-2 select-none">
            <FolderOpen size={12} />
            Folders
          </p>
          <div className="flex flex-col gap-1">
            {availableFolders.map(folder => (
              <label
                key={folder}
                className="flex items-center gap-2 text-sm text-black cursor-pointer
                           py-0.5 hover:text-black/70 transition-colors select-none"
              >
                <input
                  type="checkbox"
                  checked={selectedFolders.includes(folder)}
                  onChange={e => onFolderToggle(folder, e.target.checked)}
                  className="accent-black w-3.5 h-3.5 rounded cursor-pointer"
                />
                <span className="truncate">{folder}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      {/* ── Conversation history ─────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-2 py-3">
        {sessions.length === 0 ? (
          <p className="text-xs text-dim text-center mt-4 select-none">No conversations yet</p>
        ) : (
          <div className="flex flex-col gap-0.5">
            <p className="flex items-center gap-1.5 text-xs font-medium uppercase tracking-widest text-muted mb-2 px-2 select-none">
              <MessageSquare size={12} />
              History
            </p>
            {sessions.map(session => (
              <SessionRow
                key={session.id}
                session={session}
                isActive={session.id === activeId}
                onSelect={() => onSelectSession(session.id)}
                onDelete={() => onDeleteSession(session.id)}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}

// ── Session row ───────────────────────────────────────────────────────────────

function SessionRow({
  session,
  isActive,
  onSelect,
  onDelete,
}: {
  session:  Session
  isActive: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onSelect}
      onKeyDown={e => e.key === 'Enter' && onSelect()}
      className={clsx(
        'group flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer',
        'transition-colors text-sm',
        isActive
          ? 'bg-raised text-black font-medium'
          : 'text-muted hover:bg-raised hover:text-black',
      )}
    >
      <span className="truncate flex-1">{session.title || 'New Chat'}</span>

      {/* Delete button — only visible on hover */}
      <button
        onClick={e => { e.stopPropagation(); onDelete() }}
        className="opacity-0 group-hover:opacity-100 ml-1 p-1 rounded
                   hover:text-black text-muted transition-all shrink-0"
        title="Delete conversation"
      >
        <Trash2 size={13} />
      </button>
    </div>
  )
}
