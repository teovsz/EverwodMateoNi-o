import os
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from faq_common import DATA_DIR, get_db_connection, load_json, load_json_lines, normalize_text, save_json
from faq_models import SuggestionResponse, SuggestionSummary

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Everwod FAQ Suggestion Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "all-MiniLM-L6-v2"

FAQ_LLM_MODEL = os.getenv("FAQ_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FAQ_LLM_ENABLED = os.getenv("FAQ_LLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

FAQ_CLUSTER_EPS = float(os.getenv("FAQ_CLUSTER_EPS", "0.45"))
FAQ_MIN_CLUSTER_SIZE = int(os.getenv("FAQ_MIN_CLUSTER_SIZE", "2"))

FAQ_SKIP_EXISTING = os.getenv("FAQ_SKIP_EXISTING", "true").lower() in {"1", "true", "yes", "on"}
# FIX: subimos el threshold de 0.45 a 0.82 para que solo descarte preguntas MUY similares
FAQ_DUPLICATE_THRESHOLD = float(os.getenv("FAQ_DUPLICATE_THRESHOLD", "0.82"))

EMBEDDING_MODEL: Optional[SentenceTransformer] = None
ANSWER_GENERATOR: Optional[Any] = None
MODELS_READY = False

SUGGESTIONS_PATH = DATA_DIR / "faq_suggestions.json"
CONVERSATIONS_PATH = DATA_DIR / "conversations.jsonl"


@app.on_event("startup")
def startup_event() -> None:
    load_models()


def load_models() -> None:
    global EMBEDDING_MODEL, ANSWER_GENERATOR, MODELS_READY
    if MODELS_READY:
        return
    if EMBEDDING_MODEL is None:
        print(f"Cargando modelo de embeddings: {MODEL_NAME}...")
        EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    if FAQ_LLM_ENABLED and ANSWER_GENERATOR is None:
        try:
            print(f"Cargando LLM local: {FAQ_LLM_MODEL}...")
            tokenizer = AutoTokenizer.from_pretrained(FAQ_LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(FAQ_LLM_MODEL)
            ANSWER_GENERATOR = pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception as exc:
            print(f"No se pudo cargar {FAQ_LLM_MODEL}. Se usara respuesta historica. Error: {exc}")
            ANSWER_GENERATOR = None
    MODELS_READY = True


def load_conversation_pairs() -> List[Dict[str, str]]:
    if not CONVERSATIONS_PATH.exists():
        raise FileNotFoundError(f"Conversation file not found: {CONVERSATIONS_PATH}")
    return load_json_lines(CONVERSATIONS_PATH)


def build_protected_terms(company_name: Optional[str]) -> List[str]:
    if not company_name:
        return []
    terms = [company_name, company_name.lower()]
    for word in company_name.split():
        if len(word) > 2:
            terms.append(word)
            terms.append(word.lower())
    return list(set(terms))


def redact_personal_data(text: str, protected_terms: Optional[List[str]] = None) -> str:
    text = normalize_text(text)
    if not text:
        return ""

    protected_values: Dict[str, str] = {}
    for index, term in enumerate(protected_terms or []):
        clean_term = normalize_text(term)
        if clean_term:
            token = f"__PROTECTED_{index}__"
            protected_values[token] = clean_term
            text = re.sub(re.escape(clean_term), token, text, flags=re.IGNORECASE)

    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[correo]", text)
    text = re.sub(r"\b(?:\+?\d[\s-]?){7,}\b", "[telefono]", text)
    text = re.sub(
        r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?:\s+o\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})+\b",
        "[personas]",
        text,
    )
    text = re.sub(
        r"(?i)(hola|buenos dias|buenos días|buenas tardes|buenas noches),?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}",
        lambda match: match.group(1),
        text,
    )
    text = re.sub(
        r"(?i)(gracias por escribir(?:nos)?),?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,3}",
        r"\1",
        text,
    )
    text = re.sub(
        r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,})+\b",
        "[persona]",
        text,
    )
    text = re.sub(
        r"\b(para|de|a|por|con)\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{2,}\b",
        r"\1 [persona]",
        text,
    )
    for token, value in protected_values.items():
        text = text.replace(token, value)
    return normalize_text(text)


def is_good_faq_candidate(text: str) -> bool:
    """Filtro balanceado: descarta saludos triviales pero acepta preguntas reales de negocio."""
    text = normalize_text(text)
    if not text:
        return False

    lowered = text.lower().strip(" ¿?¡!.,;:")
    word_count = len(text.split())

    # Demasiado corto o demasiado largo
    if word_count < 2 or len(text) > 350:
        return False

    # Descartar saludos y respuestas triviales de 1-3 palabras
    trivial = {
        "hola", "buenas", "buenos dias", "buenos días", "buenas tardes",
        "buenas noches", "si", "sí", "no", "ok", "dale", "gracias",
        "muchas gracias", "de nada", "hasta luego", "adios", "adiós",
    }
    if lowered in trivial:
        return False

    # Descartar mensajes que solo tienen correos o teléfonos
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return False
    if re.search(r"\b\d(?:[\s-]?\d){6,}\b", text):
        return False

    # Con 3+ palabras es suficiente para ser candidato
    return True


def company_key(item: Dict[str, str]) -> str:
    return normalize_text(str(item.get("company_id") or item.get("workspace_id") or "unknown"))


def most_common_answer(answers: List[str], protected_terms: Optional[List[str]] = None) -> str:
    clean_answers = [
        redact_personal_data(answer, protected_terms=protected_terms)
        for answer in answers
        if redact_personal_data(answer, protected_terms=protected_terms)
    ]
    if not clean_answers:
        return "Respuesta sugerida basada en el contexto de la conversación."
    return max(set(clean_answers), key=clean_answers.count)


def load_existing_faqs_by_company() -> Dict[str, List[str]]:
    if not FAQ_SKIP_EXISTING:
        return {}

    query = (
        "SELECT a.workspace_id, af.question "
        "FROM agent_faqs af "
        "JOIN agents a ON a.id = af.agent_id "
        "WHERE af.deleted_at IS NULL AND af.question IS NOT NULL"
    )

    faqs_by_company: Dict[str, List[str]] = defaultdict(list)
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                for workspace_id, question in cursor.fetchall():
                    clean_question = normalize_text(question)
                    if clean_question:
                        faqs_by_company[str(workspace_id)].append(clean_question)
    except Exception as exc:
        print(f"No se pudieron cargar FAQs existentes. Se generaran sugerencias sin deduplicar. Error: {exc}")

    return faqs_by_company


def is_existing_faq(question: str, existing_questions: List[str]) -> bool:
    if not existing_questions:
        return False

    load_models()
    texts = [question, *existing_questions]
    embeddings = EMBEDDING_MODEL.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    similarities = embeddings[1:] @ embeddings[0]
    return bool(np.max(similarities) >= FAQ_DUPLICATE_THRESHOLD)


def generate_answer(question: str, examples: List[str], historical_answers: List[str], company_name: Optional[str]) -> str:
    protected_terms = build_protected_terms(company_name)
    fallback = most_common_answer(historical_answers, protected_terms=protected_terms)
    if not ANSWER_GENERATOR:
        return fallback

    clean_answers = [
        redact_personal_data(answer, protected_terms=protected_terms)
        for answer in historical_answers
        if redact_personal_data(answer, protected_terms=protected_terms)
    ]
    clean_examples = [
        redact_personal_data(example, protected_terms=protected_terms)
        for example in examples
        if redact_personal_data(example, protected_terms=protected_terms)
    ]
    answer_context = "\n".join(f"- {answer}" for answer in clean_answers[:4])
    example_context = "\n".join(f"- {example}" for example in clean_examples[:5])
    company_context = company_name or "esta empresa"

    messages = [
        {
            "role": "system",
            "content": (
                "Eres un asistente que redacta respuestas de FAQ en español. "
                "Usa solo la evidencia entregada, no mezcles empresas ni inventes datos. "
                "No incluyas nombres, telefonos, correos ni datos personales de clientes. "
                "Si falta informacion, responde de forma prudente indicando que se debe confirmar con el equipo."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Empresa: {company_context}\n"
                f"Pregunta FAQ propuesta: {question}\n\n"
                f"Preguntas reales similares:\n{example_context}\n\n"
                f"Respuestas historicas del asistente:\n{answer_context}\n\n"
                "Redacta una respuesta final corta, clara y util para una FAQ. "
                "No menciones que analizaste conversaciones."
            ),
        },
    ]

    tokenizer = ANSWER_GENERATOR.tokenizer
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generated = ANSWER_GENERATOR(
        prompt,
        max_new_tokens=180,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    answer = redact_personal_data(generated[0].get("generated_text", ""), protected_terms=protected_terms)
    return answer or fallback


def build_company_suggestions(
    conversations: List[Dict[str, str]],
    existing_questions: Optional[List[str]] = None,
) -> tuple[List[SuggestionResponse], Optional[float]]:
    load_models()
    existing_questions = existing_questions or []

    valid_items = []
    for item in conversations:
        user_text = normalize_text(item.get("user_text", ""))
        if user_text and is_good_faq_candidate(user_text):
            valid_items.append({**item, "user_text": user_text})

    if len(valid_items) < 2:
        return [], None

    user_texts = [item["user_text"] for item in valid_items]

    embeddings = EMBEDDING_MODEL.encode(
        user_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # FIX: eps y min_samples dinámicos según el volumen de la empresa
    effective_eps = FAQ_CLUSTER_EPS if len(valid_items) >= 30 else min(FAQ_CLUSTER_EPS + 0.10, 0.58)
    effective_min = max(2, min(FAQ_MIN_CLUSTER_SIZE, len(valid_items) // 4))

    model = DBSCAN(eps=effective_eps, min_samples=effective_min, metric="cosine")
    labels = model.fit_predict(embeddings)

    suggestions: List[SuggestionResponse] = []
    cluster_groups: Dict[int, List[int]] = {}

    for index, label in enumerate(labels):
        if label != -1:
            cluster_groups.setdefault(int(label), []).append(index)

    # FIX: si DBSCAN no formó ningún cluster (todos son ruido), agrupamos por similitud manual
    if not cluster_groups:
        print(f"   [WARN] DBSCAN no formó clusters. Aplicando agrupamiento de emergencia con eps=0.60")
        model2 = DBSCAN(eps=0.60, min_samples=2, metric="cosine")
        labels2 = model2.fit_predict(embeddings)
        for index, label in enumerate(labels2):
            if label != -1:
                cluster_groups.setdefault(int(label), []).append(index)

    if not cluster_groups:
        print(f"   [WARN] Sin clusters tras agrupamiento de emergencia.")
        return [], None

    for label, indices in cluster_groups.items():
        center = np.mean(embeddings[indices], axis=0)
        center = center / np.linalg.norm(center)
        best_index = max(indices, key=lambda idx: float(np.dot(embeddings[idx], center)))
        question_text = user_texts[best_index]
        representative = valid_items[best_index]

        if is_existing_faq(question_text, existing_questions):
            continue

        protected_terms = build_protected_terms(representative.get("company_name"))

        examples = []
        for idx in indices[:5]:
            candidate = redact_personal_data(user_texts[idx], protected_terms=protected_terms)
            if candidate not in examples:
                examples.append(candidate)

        answers = [
            normalize_text(valid_items[i].get("assistant_text", ""))
            for i in indices
            if normalize_text(valid_items[i].get("assistant_text", ""))
        ]
        answer_text = generate_answer(
            question=question_text,
            examples=examples,
            historical_answers=answers,
            company_name=representative.get("company_name"),
        )

        support = round(float(np.mean([np.dot(embeddings[idx], center) for idx in indices])), 4)

        # FIX: umbral de soporte más bajo para empresas con menos volumen
        min_support = 0.65 if len(valid_items) >= 30 else 0.50
        if support < min_support:
            continue

        cluster_score = round(min(100.0, 100.0 * len(indices) / len(valid_items)), 2)
        suggestions.append(
            SuggestionResponse(
                id=str(uuid.uuid4()),
                company_id=company_key(representative),
                company_name=representative.get("company_name"),
                question=question_text,
                answer=answer_text,
                cluster_size=len(indices),
                support_examples=examples[:3],
                cluster_score=cluster_score,
            )
        )

    silhouette = None
    clean_labels = [label for label in labels if label != -1]
    clean_embeddings = embeddings[labels != -1]
    if len(set(clean_labels)) > 1 and len(clean_embeddings) > len(set(clean_labels)):
        try:
            silhouette = round(silhouette_score(clean_embeddings, clean_labels, metric="cosine"), 4)
        except Exception:
            silhouette = None

    return suggestions, silhouette


def build_suggestions(conversations: List[Dict[str, str]]) -> SuggestionSummary:
    if not conversations:
        raise ValueError("No conversation pairs available for suggestion generation.")

    conversations_by_company: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for item in conversations:
        conversations_by_company[company_key(item)].append(item)

    print("\n" + "="*50)
    print("INICIANDO GENERACION DE SUGERENCIAS")

    suggestions: List[SuggestionResponse] = []
    silhouettes = []
    total_examples = 0
    existing_faqs_by_company = load_existing_faqs_by_company()

    for current_company, items in conversations_by_company.items():
        valid_count = sum(1 for item in items if is_good_faq_candidate(item.get("user_text", "")))
        print(f"Empresa [{current_company}]: {len(items)} mensajes -> {valid_count} validos")

        total_examples += valid_count
        company_suggestions, company_silhouette = build_company_suggestions(
            items,
            existing_questions=existing_faqs_by_company.get(current_company, []),
        )

        print(f"   -> {len(company_suggestions)} sugerencias generadas")
        suggestions.extend(company_suggestions)
        if company_silhouette is not None:
            silhouettes.append(company_silhouette)

    print("="*50 + "\n")

    if not suggestions:
        raise HTTPException(
            status_code=400,
            detail="Ninguna conversación cumple el formato mínimo para armar FAQs."
        )

    silhouette = round(sum(silhouettes) / len(silhouettes), 4) if silhouettes else None
    avg_size = round(total_examples / len(suggestions), 2) if suggestions else 0

    summary = SuggestionSummary(
        company_count=len(conversations_by_company),
        cluster_count=len(suggestions),
        total_examples=total_examples,
        average_cluster_size=avg_size,
        silhouette_score=silhouette,
        suggestions=suggestions,
    )
    save_json(summary.dict(), SUGGESTIONS_PATH)
    return summary


@app.get("/health")
def health() -> dict:
    load_models()
    return {
        "status": "ok",
        "service": "suggestion",
        "embedding_model": MODEL_NAME,
        "answer_model": FAQ_LLM_MODEL if ANSWER_GENERATOR else "historical_fallback",
    }


@app.post("/suggest", response_model=SuggestionSummary)
def suggest() -> SuggestionSummary:
    conversations = load_conversation_pairs()
    return build_suggestions(conversations)


@app.get("/suggestions")
def get_suggestions() -> dict:
    if not SUGGESTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="No suggestions have been generated yet.")

    raw = load_json(SUGGESTIONS_PATH)

    all_companies = {}
    try:
        conversations = load_conversation_pairs()
        for item in conversations:
            c_id = company_key(item)
            c_name = item.get("company_name") or f"Empresa {c_id}"
            if c_id and c_id != "unknown":
                all_companies[c_id] = c_name
    except Exception:
        pass

    companies_list = [{"company_id": k, "company_name": v} for k, v in all_companies.items()]
    raw["all_companies_in_db"] = companies_list

    return raw


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("suggestion_service:app", host="127.0.0.1", port=8003, log_level="info")