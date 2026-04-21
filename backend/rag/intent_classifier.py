"""
intent_classifier.py — Route questions to the right execution path (Track 2).

Three intents:
  RAG    — open-ended document questions (existing pipeline)
  SQL    — structured/aggregation questions about patient data
  HYBRID — needs both: find patients via SQL, then pull context via RAG

The classifier runs one llama3.2:3b call per question. It sits at the very
top of the pipeline, before complexity routing or HyDE expansion.
"""

import logging

from models.ollama import generate_worker

logger = logging.getLogger(__name__)

# Valid intent labels — anything else falls back to RAG
_VALID_INTENTS = {"RAG", "SQL", "HYBRID"}

_CLASSIFICATION_PROMPT = """\
You are a query router for a medical document system. The system has two data sources:

1. PATIENT TABLE — structured records with: full_name, date_of_birth, gender, \
city, state, insurance_provider, insurance_id, icd10_codes (array), \
diagnoses (array), medications (array), medical_history (text).

2. DOCUMENT STORE — full text of thousands of medical PDFs, searchable by \
semantic similarity.

Classify the user's question into exactly one category:

SQL — The question can be answered entirely from the patient table. \
Examples: counting patients, filtering by age/gender/medication/diagnosis/ICD code, \
listing patients matching criteria, aggregations (average age, most common diagnosis).

HYBRID — The question needs BOTH the patient table AND document text. \
Examples: "summarise treatment plans for patients on Metformin" (SQL finds the \
patients, documents provide the treatment details), "compare notes for patients \
over 65 with diabetes".

RAG — Everything else. The question is about document content, procedures, \
recommendations, policies, or specific patient notes. Examples: "what does the \
doctor recommend for knee pain?", "what is the discharge plan for patient X?", \
"explain the medication interactions mentioned in the report".

Question: {question}

Reply with one word only — SQL, HYBRID, or RAG."""


def classify_intent(question: str) -> str:
    """
    Classify a question as RAG, SQL, or HYBRID.

    Returns one of: "RAG", "SQL", "HYBRID".
    Falls back to "RAG" on any error — safe default that preserves
    the existing pipeline behaviour.
    """
    prompt = _CLASSIFICATION_PROMPT.format(question=question)
    try:
        raw = generate_worker(prompt).strip().upper()
        # Extract the first valid intent from the response
        for intent in _VALID_INTENTS:
            if intent in raw:
                logger.info(f"Intent classified as: {intent}")
                return intent
        logger.warning(f"Unrecognised intent response: {raw!r} — defaulting to RAG")
        return "RAG"
    except Exception as e:
        logger.error(f"Intent classification failed: {e} — defaulting to RAG")
        return "RAG"
