import { useCallback, useEffect, useRef, useState } from 'react'
import { Search, ChevronDown, ChevronRight, FileText } from 'lucide-react'
import { clsx } from 'clsx'
import * as api from '../lib/api'
import type { PatientSummary, PatientDetail, PatientStats } from '../types'

const PAGE_SIZE = 50

// ── Helpers ──────────────────────────────────────────────────────────────────

function age(dob: string | null): string {
  if (!dob) return '—'
  const years = Math.floor((Date.now() - new Date(dob).getTime()) / 31_557_600_000)
  return `${years} y`
}

function completenessColor(pct: number): string {
  if (pct >= 90) return 'bg-green-100 text-green-700'
  if (pct >= 60) return 'bg-yellow-100 text-yellow-700'
  return 'bg-red-100 text-red-700'
}

// ── PatientBrowser ────────────────────────────────────────────────────────────

export default function PatientBrowser() {
  const [stats,      setStats]      = useState<PatientStats | null>(null)
  const [patients,   setPatients]   = useState<PatientSummary[]>([])
  const [total,      setTotal]      = useState(0)
  const [offset,     setOffset]     = useState(0)
  const [loading,    setLoading]    = useState(true)
  const [expandedId, setExpandedId] = useState<string | null>(null)

  // Filter state
  const [name,       setName]       = useState('')
  const [gender,     setGender]     = useState('')
  const [medication, setMedication] = useState('')
  const [icd10,      setIcd10]      = useState('')

  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    api.getPatientStats().then(setStats).catch(() => {})
  }, [])

  const load = useCallback((
    filters: { name: string; gender: string; medication: string; icd10: string },
    page: number,
  ) => {
    setLoading(true)
    api.listPatients({
      name:       filters.name       || undefined,
      gender:     filters.gender     || undefined,
      medication: filters.medication || undefined,
      icd10:      filters.icd10      || undefined,
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    })
      .then(res => {
        setPatients(res.patients)
        setTotal(res.total)
      })
      .catch(() => setPatients([]))
      .finally(() => setLoading(false))
  }, [])

  // Debounce text filters, immediate for dropdowns
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => {
      setOffset(0)
      load({ name, gender, medication, icd10 }, 0)
    }, 300)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [name, gender, medication, icd10, load])

  const handlePage = (dir: 1 | -1) => {
    const next = offset + dir * PAGE_SIZE
    setOffset(next)
    load({ name, gender, medication, icd10 }, next / PAGE_SIZE)
  }

  const totalPages = Math.ceil(total / PAGE_SIZE)
  const currentPage = Math.floor(offset / PAGE_SIZE)

  return (
    <div className="flex flex-col h-full bg-surface">

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="px-6 pt-6 pb-4 border-b border-border">
        <h1 className="text-lg font-semibold tracking-tight">Patient Records</h1>
        {stats && (
          <p className="text-xs text-muted mt-1">
            {stats.total_patients.toLocaleString()} patients extracted
          </p>
        )}
      </div>

      {/* ── Field completeness badges ──────────────────────────────────────── */}
      {stats && stats.total_patients > 0 && (
        <div className="flex flex-wrap gap-1.5 px-6 pt-3 pb-2">
          {Object.entries(stats.field_completeness).map(([field, count]) => {
            const pct = Math.round((count / stats.total_patients) * 100)
            return (
              <span
                key={field}
                className={clsx('px-2 py-0.5 rounded-full text-xs', completenessColor(pct))}
                title={`${count} / ${stats.total_patients} patients have ${field}`}
              >
                {field.replace(/_/g, ' ')} {pct}%
              </span>
            )
          })}
        </div>
      )}

      {/* ── Filters ────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap gap-2 px-6 py-3">
        {/* Name search */}
        <div className="relative flex-1 min-w-[160px]">
          <Search size={13} className="absolute left-3 top-1/2 -translate-y-1/2 text-dim" />
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Search by name..."
            className="w-full pl-8 pr-3 py-1.5 text-sm bg-raised border border-border rounded-lg
                       outline-none focus:ring-1 focus:ring-black/20 placeholder:text-dim"
          />
        </div>

        {/* Gender dropdown */}
        <select
          value={gender}
          onChange={e => setGender(e.target.value)}
          className="px-3 py-1.5 text-sm bg-raised border border-border rounded-lg
                     outline-none focus:ring-1 focus:ring-black/20 text-black"
        >
          <option value="">Any gender</option>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>

        {/* Medication filter */}
        <div className="relative">
          <input
            type="text"
            value={medication}
            onChange={e => setMedication(e.target.value)}
            placeholder="Medication..."
            className="pl-3 pr-3 py-1.5 text-sm bg-raised border border-border rounded-lg
                       outline-none focus:ring-1 focus:ring-black/20 placeholder:text-dim w-36"
          />
        </div>

        {/* ICD-10 filter */}
        <div className="relative">
          <input
            type="text"
            value={icd10}
            onChange={e => setIcd10(e.target.value)}
            placeholder="ICD-10 code..."
            className="pl-3 pr-3 py-1.5 text-sm bg-raised border border-border rounded-lg
                       outline-none focus:ring-1 focus:ring-black/20 placeholder:text-dim w-36"
          />
        </div>
      </div>

      {/* ── Patient list ───────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-4 pb-4">
        {loading ? (
          <p className="text-sm text-muted text-center mt-8">Loading...</p>
        ) : patients.length === 0 ? (
          <div className="text-center mt-8">
            <p className="text-sm text-muted">No patients found</p>
            {stats?.total_patients === 0 && (
              <p className="text-xs text-dim mt-1">
                Ingest patient folders to populate records
              </p>
            )}
          </div>
        ) : (
          <div className="flex flex-col gap-1">
            {patients.map(patient => (
              <PatientRow
                key={patient.patient_id}
                patient={patient}
                isExpanded={expandedId === patient.patient_id}
                onToggle={() => setExpandedId(prev =>
                  prev === patient.patient_id ? null : patient.patient_id
                )}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── Pagination ─────────────────────────────────────────────────────── */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-6 py-3 border-t border-border text-sm text-muted">
          <span>
            {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total.toLocaleString()}
          </span>
          <div className="flex gap-2">
            <button
              disabled={currentPage === 0}
              onClick={() => handlePage(-1)}
              className="px-3 py-1 rounded border border-border disabled:opacity-30
                         hover:bg-raised transition-colors"
            >
              Prev
            </button>
            <button
              disabled={currentPage >= totalPages - 1}
              onClick={() => handlePage(1)}
              className="px-3 py-1 rounded border border-border disabled:opacity-30
                         hover:bg-raised transition-colors"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}


// ── Patient row ───────────────────────────────────────────────────────────────

function PatientRow({
  patient,
  isExpanded,
  onToggle,
}: {
  patient:    PatientSummary
  isExpanded: boolean
  onToggle:   () => void
}) {
  const [detail,     setDetail]     = useState<PatientDetail | null>(null)
  const [loadingDet, setLoadingDet] = useState(false)

  useEffect(() => {
    if (!isExpanded || detail) return
    setLoadingDet(true)
    api.getPatient(patient.patient_id)
      .then(setDetail)
      .catch(() => {})
      .finally(() => setLoadingDet(false))
  }, [isExpanded, patient.patient_id, detail])

  return (
    <div className="border border-border rounded-lg overflow-hidden">
      {/* ── Row header ──────────────────────────────────────────────────── */}
      <button
        onClick={onToggle}
        className={clsx(
          'w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors',
          isExpanded ? 'bg-raised' : 'hover:bg-raised/50',
        )}
      >
        {isExpanded
          ? <ChevronDown size={14} className="text-muted shrink-0" />
          : <ChevronRight size={14} className="text-muted shrink-0" />
        }

        <span className="text-sm font-medium flex-1 truncate">
          {patient.full_name || <span className="text-dim italic">Unknown</span>}
        </span>

        <div className="flex items-center gap-3 shrink-0 text-xs text-muted">
          {patient.gender && <span>{patient.gender}</span>}
          {patient.date_of_birth && <span>{age(patient.date_of_birth)}</span>}
          {(patient.city || patient.state) && (
            <span>{[patient.city, patient.state].filter(Boolean).join(', ')}</span>
          )}
          {patient.medications.length > 0 && (
            <span className="bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded text-[10px]">
              {patient.medications.length} med{patient.medications.length !== 1 ? 's' : ''}
            </span>
          )}
          {patient.diagnoses.length > 0 && (
            <span className="bg-purple-50 text-purple-600 px-1.5 py-0.5 rounded text-[10px]">
              {patient.diagnoses.length} dx
            </span>
          )}
        </div>
      </button>

      {/* ── Expanded detail ──────────────────────────────────────────────── */}
      {isExpanded && (
        <div className="px-4 py-4 border-t border-border bg-white">
          {loadingDet ? (
            <p className="text-xs text-dim">Loading...</p>
          ) : detail ? (
            <PatientDetailPanel detail={detail} />
          ) : (
            <p className="text-xs text-dim">Failed to load detail</p>
          )}
        </div>
      )}
    </div>
  )
}


// ── Patient detail panel ──────────────────────────────────────────────────────

function PatientDetailPanel({ detail }: { detail: PatientDetail }) {
  const fields: Array<[string, string | null]> = [
    ['Date of birth', detail.date_of_birth
      ? `${detail.date_of_birth} (${age(detail.date_of_birth)})`
      : null],
    ['Gender',             detail.gender],
    ['City',               detail.city],
    ['State',              detail.state],
    ['Insurance provider', detail.insurance_provider],
    ['Insurance ID',       detail.insurance_id],
  ]

  return (
    <div className="flex flex-col gap-4">
      {/* Scalar fields */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-1.5">
        {fields.map(([label, value]) => (
          <div key={label}>
            <p className="text-[10px] font-medium uppercase tracking-widest text-muted">{label}</p>
            <p className="text-sm">{value || <span className="text-dim italic">—</span>}</p>
          </div>
        ))}
      </div>

      {/* Array fields */}
      {detail.medications.length > 0 && (
        <FieldChips label="Medications" items={detail.medications} color="bg-blue-50 text-blue-700" />
      )}
      {detail.diagnoses.length > 0 && (
        <FieldChips label="Diagnoses" items={detail.diagnoses} color="bg-purple-50 text-purple-700" />
      )}
      {detail.icd10_codes.length > 0 && (
        <FieldChips label="ICD-10 codes" items={detail.icd10_codes} color="bg-gray-100 text-gray-600" />
      )}

      {/* Medical history */}
      {detail.medical_history && (
        <div>
          <p className="text-[10px] font-medium uppercase tracking-widest text-muted mb-1">
            Medical history
          </p>
          <p className="text-xs text-black leading-relaxed">{detail.medical_history}</p>
        </div>
      )}

      {/* Provenance */}
      {detail.provenance.length > 0 && (
        <div>
          <p className="text-[10px] font-medium uppercase tracking-widest text-muted mb-2">
            Source provenance
          </p>
          <div className="flex flex-col gap-1">
            {detail.provenance.map((p, i) => (
              <div key={i} className="flex items-start gap-2 text-xs py-0.5">
                <FileText size={11} className="text-muted mt-0.5 shrink-0" />
                <span className="text-muted w-28 shrink-0">{p.field_name}</span>
                <span className="truncate flex-1">{p.field_value}</span>
                <span className="text-dim shrink-0 truncate max-w-[180px]">
                  {p.source_file}, p.{p.source_page}
                  {p.confidence < 1 && (
                    <span className="ml-1 text-[10px] opacity-60">
                      ({Math.round(p.confidence * 100)}%)
                    </span>
                  )}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}


// ── Chip list helper ──────────────────────────────────────────────────────────

function FieldChips({ label, items, color }: { label: string; items: string[]; color: string }) {
  return (
    <div>
      <p className="text-[10px] font-medium uppercase tracking-widest text-muted mb-1.5">{label}</p>
      <div className="flex flex-wrap gap-1">
        {items.map((item, i) => (
          <span key={i} className={clsx('px-2 py-0.5 rounded text-xs', color)}>{item}</span>
        ))}
      </div>
    </div>
  )
}
