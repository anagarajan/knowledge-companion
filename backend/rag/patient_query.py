"""
patient_query.py — Natural language → structured patient queries (Track 2).

Security model:
  The LLM never writes SQL. It outputs a JSON filter spec, and this module
  builds a parameterised query from it. This eliminates SQL injection by
  construction — the LLM controls filter values, not query structure.

Flow:
  1. parse_patient_question()  — LLM → JSON filter spec
  2. build_patient_query()     — filter spec → parameterised SQL
  3. execute_patient_query()   — run with statement_timeout + row cap
  4. format_patient_results()  — rows → markdown answer (streamed)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

from config import PATIENT_QUERY_MAX_ROWS, PATIENT_QUERY_TIMEOUT_MS
from db.connection import db_conn
from models.ollama import generate_worker_json, generate_worker

logger = logging.getLogger(__name__)


# ── Filter spec schema ───────────────────────────────────────────────────────
#
# The LLM outputs JSON matching this shape. Every field is optional.
# Example for "female patients over 65 on Metformin":
#   {
#     "filters": {
#       "gender": "Female",
#       "age_min": 65,
#       "medications_contain": ["Metformin"]
#     },
#     "select": ["full_name", "date_of_birth", "medications"],
#     "aggregation": null,
#     "limit": 20
#   }

# Columns the LLM is allowed to request in "select"
_ALLOWED_COLUMNS = frozenset({
    "patient_id", "full_name", "date_of_birth", "gender",
    "city", "state", "insurance_provider", "insurance_id",
    "icd10_codes", "diagnoses", "medications", "medical_history",
})

# Aggregation functions the LLM can request
_ALLOWED_AGGREGATIONS = frozenset({
    "count", "count_distinct", "list",
})

_FILTER_PROMPT = """\
You are a query parser for a medical patient database. Convert the user's \
natural language question into a structured JSON filter.

The patient table has these columns:
- full_name (text)
- date_of_birth (date) — use age_min / age_max filters instead of raw dates
- gender (text: "Male", "Female", or null)
- city (text)
- state (text)
- insurance_provider (text)
- insurance_id (text)
- icd10_codes (array of strings, e.g. ["E11.9", "I10"])
- diagnoses (array of strings, e.g. ["Type 2 Diabetes", "Hypertension"])
- medications (array of strings, e.g. ["Metformin", "Lisinopril"])
- medical_history (text — free text summary)

Return a JSON object with these keys:
{{
  "filters": {{
    "gender": "Male" or "Female" or null,
    "age_min": integer or null,
    "age_max": integer or null,
    "city": "exact city" or null,
    "state": "exact state" or null,
    "insurance_provider": "provider name" or null,
    "icd10_codes_contain": ["code1", "code2"] or null,
    "diagnoses_contain": ["term1"] or null,
    "medications_contain": ["drug1"] or null,
    "name_search": "partial name" or null
  }},
  "select": ["column1", "column2"],
  "aggregation": "count" or "count_distinct" or "list" or null,
  "limit": integer (default 20, max 100)
}}

Rules:
- Only include filters that the question actually mentions
- For "how many" questions, set aggregation to "count"
- For "list all" questions, set aggregation to "list"
- select should include the columns relevant to answering the question
- Always include "full_name" in select unless it's a pure count
- Set null for any filter not mentioned in the question

Question: {question}"""


@dataclass(frozen=True)
class PatientFilterSpec:
    """Parsed filter specification from the LLM."""
    filters: dict[str, Any]
    select: list[str]
    aggregation: str | None
    limit: int


@dataclass(frozen=True)
class PatientQueryResult:
    """Result of executing a patient query."""
    rows: list[dict]
    total_count: int
    query_description: str  # human-readable description of what was queried
    columns: list[str]


# ── Step 1: Parse question → filter spec ─────────────────────────────────────


def parse_patient_question(question: str) -> PatientFilterSpec:
    """
    Ask the LLM to convert a natural language question into a structured
    filter spec. Uses format=json to enforce valid JSON output.
    """
    prompt = _FILTER_PROMPT.format(question=question)
    raw = generate_worker_json(prompt)

    # Extract and validate filters
    filters = raw.get("filters", {})
    if not isinstance(filters, dict):
        filters = {}

    # Validate select columns — reject anything not in the whitelist
    raw_select = raw.get("select", ["full_name"])
    select = [c for c in raw_select if c in _ALLOWED_COLUMNS]
    if not select:
        select = ["full_name"]

    # Validate aggregation
    aggregation = raw.get("aggregation")
    if aggregation and aggregation not in _ALLOWED_AGGREGATIONS:
        aggregation = None

    # Validate limit
    limit = raw.get("limit", 20)
    if not isinstance(limit, int) or limit < 1:
        limit = 20
    limit = min(limit, PATIENT_QUERY_MAX_ROWS)

    return PatientFilterSpec(
        filters=filters,
        select=select,
        aggregation=aggregation,
        limit=limit,
    )


# ── Step 2: Filter spec → parameterised SQL ─────────────────────────────────


def build_patient_query(spec: PatientFilterSpec) -> tuple[str, list]:
    """
    Convert a PatientFilterSpec into a parameterised SQL query.

    Returns (sql_string, params_list). The SQL uses %s placeholders
    compatible with psycopg2.

    No user-controlled strings enter the SQL template — only parameterised
    values. Column names come from the validated whitelist.
    """
    conditions: list[str] = []
    params: list = []

    f = spec.filters

    # ── Scalar filters ───────────────────────────────────────────────────
    if f.get("gender"):
        conditions.append("gender ILIKE %s")
        params.append(f["gender"])

    if f.get("age_min") is not None:
        conditions.append(
            "date_of_birth <= CURRENT_DATE - make_interval(years => %s)"
        )
        params.append(int(f["age_min"]))

    if f.get("age_max") is not None:
        conditions.append(
            "date_of_birth >= CURRENT_DATE - make_interval(years => %s)"
        )
        params.append(int(f["age_max"]))

    if f.get("city"):
        conditions.append("city ILIKE %s")
        params.append(f["city"])

    if f.get("state"):
        conditions.append("state ILIKE %s")
        params.append(f["state"])

    if f.get("insurance_provider"):
        conditions.append("insurance_provider ILIKE %s")
        params.append(f["insurance_provider"])

    if f.get("name_search"):
        conditions.append("full_name ILIKE %s")
        params.append(f"%{f['name_search']}%")

    # ── JSONB array containment filters (use GIN indexes) ────────────────
    for field, filter_key in (
        ("icd10_codes",  "icd10_codes_contain"),
        ("diagnoses",    "diagnoses_contain"),
        ("medications",  "medications_contain"),
    ):
        values = f.get(filter_key)
        if values and isinstance(values, list):
            # @> checks if the array contains all specified elements
            conditions.append(f"{field} @> %s::jsonb")
            params.append(json.dumps(values))

    # ── Build SELECT clause ──────────────────────────────────────────────
    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    if spec.aggregation == "count":
        sql = f"SELECT COUNT(*) AS total FROM patients WHERE {where_clause}"
        return sql, params

    if spec.aggregation == "count_distinct":
        # Count distinct on the first selected column
        col = spec.select[0]
        sql = f"SELECT COUNT(DISTINCT {col}) AS total FROM patients WHERE {where_clause}"
        return sql, params

    # Regular select or "list" aggregation
    columns = ", ".join(spec.select)
    sql = (
        f"SELECT {columns} FROM patients "
        f"WHERE {where_clause} "
        f"ORDER BY full_name "
        f"LIMIT %s"
    )
    params.append(spec.limit)

    return sql, params


# ── Step 3: Execute with safety limits ───────────────────────────────────────


def execute_patient_query(spec: PatientFilterSpec) -> PatientQueryResult:
    """
    Build and execute the patient query with statement_timeout and row cap.
    """
    sql, params = build_patient_query(spec)
    logger.info(f"Patient query: {sql}")
    logger.debug(f"Params: {params}")

    with db_conn() as conn:
        with conn.cursor() as cur:
            # Set per-query timeout to prevent runaway queries
            cur.execute(
                "SET LOCAL statement_timeout = %s",
                (PATIENT_QUERY_TIMEOUT_MS,),
            )

            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

            # For non-count queries, also get the total matching count
            if spec.aggregation == "count":
                total = rows[0]["total"] if rows else 0
                return PatientQueryResult(
                    rows=rows,
                    total_count=total,
                    query_description=_describe_filters(spec),
                    columns=["total"],
                )

            # Get total count for pagination context
            count_sql, count_params = build_patient_query(
                PatientFilterSpec(
                    filters=spec.filters,
                    select=spec.select,
                    aggregation="count",
                    limit=spec.limit,
                )
            )
            cur.execute(count_sql, count_params)
            total = cur.fetchone()["total"]

            columns = spec.select
            return PatientQueryResult(
                rows=rows,
                total_count=total,
                query_description=_describe_filters(spec),
                columns=columns,
            )


# ── Step 4: Format results → streamed markdown ──────────────────────────────


def format_patient_results(
    question: str,
    result: PatientQueryResult,
) -> Generator[str, None, None]:
    """
    Convert query results into a natural-language markdown answer.

    Uses llama3.2:3b to produce a concise summary rather than dumping
    raw table data. The LLM sees the structured results and the original
    question, then writes a human-friendly response.
    """
    if result.total_count == 0:
        yield (
            "No patients found matching those criteria. "
            "Try broadening your search or checking the filter terms."
        )
        return

    # For count queries, answer directly — no LLM needed
    if result.columns == ["total"]:
        yield f"**{result.total_count}** patients match: {result.query_description}"
        return

    # Build a data summary for the LLM
    # Cap at 30 rows in the prompt to stay within context limits
    display_rows = result.rows[:30]
    rows_text = _format_rows_as_text(display_rows, result.columns)

    showing = len(display_rows)
    total = result.total_count
    count_note = (
        f"Showing {showing} of {total} matching patients."
        if total > showing
        else f"{total} patients found."
    )

    prompt = (
        "You are a medical data assistant. The user asked a question about patient records. "
        "Answer their question using the data below. Format your response with markdown.\n\n"
        "Rules:\n"
        "1. Be concise and directly answer the question.\n"
        "2. Use a markdown table if listing multiple patients.\n"
        "3. For dates of birth, also show the age in parentheses.\n"
        "4. If the data is truncated, mention the total count.\n"
        "5. Do NOT add information beyond what the data shows.\n\n"
        f"Question: {question}\n\n"
        f"{count_note}\n\n"
        f"Data:\n{rows_text}"
    )

    # Use generate_worker (non-streaming) then yield the full response.
    # Patient queries return structured data so responses are short — streaming
    # token by token adds latency from the LLM call overhead without benefit.
    answer = generate_worker(prompt)
    yield answer


# ── Helpers ──────────────────────────────────────────────────────────────────


def _describe_filters(spec: PatientFilterSpec) -> str:
    """Human-readable description of active filters."""
    parts: list[str] = []
    f = spec.filters

    if f.get("gender"):
        parts.append(f"gender = {f['gender']}")
    if f.get("age_min") is not None:
        parts.append(f"age >= {f['age_min']}")
    if f.get("age_max") is not None:
        parts.append(f"age <= {f['age_max']}")
    if f.get("city"):
        parts.append(f"city = {f['city']}")
    if f.get("state"):
        parts.append(f"state = {f['state']}")
    if f.get("insurance_provider"):
        parts.append(f"insurance = {f['insurance_provider']}")
    if f.get("name_search"):
        parts.append(f"name contains '{f['name_search']}'")
    for key, label in (
        ("icd10_codes_contain", "ICD-10 codes include"),
        ("diagnoses_contain", "diagnoses include"),
        ("medications_contain", "medications include"),
    ):
        if f.get(key):
            parts.append(f"{label} {f[key]}")

    return ", ".join(parts) if parts else "all patients"


def _format_rows_as_text(rows: list[dict], columns: list[str]) -> str:
    """Format rows as a simple text table for the LLM prompt."""
    if not rows:
        return "(no data)"

    lines: list[str] = []
    # Header
    lines.append(" | ".join(columns))
    lines.append(" | ".join("---" for _ in columns))
    # Rows
    for row in rows:
        values = []
        for col in columns:
            val = row.get(col)
            if val is None:
                values.append("—")
            elif isinstance(val, list):
                values.append(", ".join(str(v) for v in val))
            else:
                values.append(str(val))
        lines.append(" | ".join(values))

    return "\n".join(lines)


# ── Public convenience: one-shot query ───────────────────────────────────────


def answer_patient_question(question: str) -> tuple[PatientQueryResult, Generator[str, None, None]]:
    """
    End-to-end: parse → execute → format.
    Returns (result_metadata, token_stream) so the caller can build
    sources and confidence alongside the streamed answer.
    """
    spec = parse_patient_question(question)
    result = execute_patient_query(spec)
    token_stream = format_patient_results(question, result)
    return result, token_stream
