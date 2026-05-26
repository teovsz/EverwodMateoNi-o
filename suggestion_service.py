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

# Aplicacion FastAPI del servicio de sugerencias.
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Everwod FAQ Suggestion Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mismo modelo usado para representar semanticamente las preguntas de usuarios.
MODEL_NAME = "all-MiniLM-L6-v2"

# Modelo local pequeno para redactar respuestas sugeridas. Se puede cambiar en .env.
FAQ_LLM_MODEL = os.getenv("FAQ_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
FAQ_LLM_ENABLED = os.getenv("FAQ_LLM_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
FAQ_CLUSTER_EPS = float(os.getenv("FAQ_CLUSTER_EPS", "0.34"))
FAQ_MIN_CLUSTER_SIZE = int(os.getenv("FAQ_MIN_CLUSTER_SIZE", "3"))
FAQ_SKIP_EXISTING = os.getenv("FAQ_SKIP_EXISTING", "true").lower() in {"1", "true", "yes", "on"}
FAQ_DUPLICATE_THRESHOLD = float(os.getenv("FAQ_DUPLICATE_THRESHOLD", "0.78"))

# Se cargan bajo demanda para reutilizarlos durante toda la vida del servicio.
EMBEDDING_MODEL: Optional[SentenceTransformer] = None
ANSWER_GENERATOR: Optional[Any] = None
MODELS_READY = False

# Archivos de entrada y salida del servicio.
SUGGESTIONS_PATH = DATA_DIR / "faq_suggestions.json"
CONVERSATIONS_PATH = DATA_DIR / "conversations.jsonl"


@app.on_event("startup")
def startup_event() -> None:
    """Carga los modelos locales cuando arranca el servicio."""
    load_models()


def load_models() -> None:
    """Carga modelos si aun no estan listos; tambien sirve para ejecuciones por scheduler."""
    global EMBEDDING_MODEL, ANSWER_GENERATOR, MODELS_READY
    if MODELS_READY:
        return
    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = SentenceTransformer(MODEL_NAME)
    if FAQ_LLM_ENABLED:
        try:
            tokenizer = AutoTokenizer.from_pretrained(FAQ_LLM_MODEL)
            model = AutoModelForCausalLM.from_pretrained(FAQ_LLM_MODEL)
            ANSWER_GENERATOR = pipeline("text-generation", model=model, tokenizer=tokenizer)
        except Exception as exc:
            print(f"No se pudo cargar {FAQ_LLM_MODEL}. Se usara respuesta historica. Error: {exc}")
            ANSWER_GENERATOR = None
    MODELS_READY = True


def load_conversation_pairs() -> List[Dict[str, str]]:
    """Lee los pares usuario/asistente generados por el servicio de ingesta."""
    if not CONVERSATIONS_PATH.exists():
        raise FileNotFoundError(f"Conversation file not found: {CONVERSATIONS_PATH}")
    return load_json_lines(CONVERSATIONS_PATH)


def build_protected_terms(company_name: Optional[str]) -> List[str]:
    """Construye la lista de terminos protegidos para no borrar el nombre de la empresa."""
    if not company_name:
        return []
    terms = [company_name, company_name.lower()]
    # Agrega cada palabra del nombre de la empresa como termino protegido tambien.
    for word in company_name.split():
        if len(word) > 2:
            terms.append(word)
            terms.append(word.lower())
    return list(set(terms))


def redact_personal_data(text: str, protected_terms: Optional[List[str]] = None) -> str:
    """Elimina datos personales frecuentes antes de exponer o usar texto como contexto."""
    text = normalize_text(text)
    if not text:
        return ""

    protected_values: Dict[str, str] = {}
    for index, term in enumerate(protected_terms or []):
        clean_term = normalize_text(term)
        if clean_term:
            token = f"__PROTECTED_{index}__"
            protected_values[token] = clean_term
            # Reemplaza con case-insensitive para cubrir variaciones de mayusculas.
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
    """Descarta mensajes demasiado conversacionales o especificos para ser FAQ."""
    text = normalize_text(text)
    lowered = text.lower().strip(" ¿?¡!.,;:")
    word_count = len(text.split())

    if len(text) < 12 or word_count < 3:
        return False
    if len(text) > 280:
        return False
    if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text):
        return False
    if re.search(r"\b\d(?:[\s-]?\d){6,}\b", text):
        return False

    trivial_messages = {
        "hola",
        "buenas",
        "buenos dias",
        "buenos días",
        "buenas tardes",
        "buenas noches",
        "si",
        "sí",
        "no",
        "ok",
        "dale",
        "gracias",
    }
    if lowered in trivial_messages:
        return False

    non_faq_fragments = (
        "quien soy",
        "quién soy",
        "que hora",
        "qué hora",
        "que rol",
        "qué rol",
        "hora es actualmente",
        "que usuarios",
        "qué usuarios",
        "usuarios agendaron",
        "usuarios estarán",
        "usuarios estaran",
        "personas asistiran",
        "personas asistirán",
        "quien asist",
        "quién asist",
    )
    if any(fragment in lowered for fragment in non_faq_fragments):
        return False
    if "gracias" in lowered and word_count <= 7 and "?" not in text:
        return False

    has_question_signal = "?" in text or any(
        lowered.startswith(prefix)
        for prefix in (
            "como ",
            "cómo ",
            "cuanto ",
            "cuánto ",
            "donde ",
            "dónde ",
            "cuando ",
            "cuándo ",
            "puedo ",
            "quiero ",
            "quisiera ",
            "necesito ",
            "tienen ",
            "me ayudas ",
            "me puedes ",
            "me puede ",
            "me gustaria ",
            "me gustaría ",
            "seria posible ",
            "sería posible ",
        )
    )
    intent_keywords = (
        "precio",
        "precios",
        "plan",
        "planes",
        "mensualidad",
        "horario",
        "horarios",
        "ubicacion",
        "ubicación",
        "direccion",
        "dirección",
        "reservar",
        "reserva",
        "agendar",
        "clase",
        "pagar",
        "pago",
        "qr",
        "crossfit",
        "informacion",
        "información",
    )
    has_business_intent = any(keyword in lowered for keyword in intent_keywords)
    return has_question_signal and has_business_intent


def company_key(item: Dict[str, str]) -> str:
    """Devuelve el identificador de empresa preservando compatibilidad con ingestas viejas."""
    return normalize_text(str(item.get("company_id") or item.get("workspace_id") or "unknown"))


def most_common_answer(answers: List[str], protected_terms: Optional[List[str]] = None) -> str:
    """Escoge una respuesta historica como fallback cuando el LLM no esta disponible."""
    clean_answers = [
        redact_personal_data(answer, protected_terms=protected_terms)
        for answer in answers
        if redact_personal_data(answer, protected_terms=protected_terms)
    ]
    if not clean_answers:
        return "Respuesta sugerida basada en el contexto de la conversación."
    return max(set(clean_answers), key=clean_answers.count)


def load_existing_faqs_by_company() -> Dict[str, List[str]]:
    """Carga FAQs existentes por workspace para evitar sugerencias duplicadas."""
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
    """Compara semanticamente una pregunta sugerida contra FAQs ya creadas."""
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
    """Redacta una respuesta FAQ con Qwen usando solo evidencia de la misma empresa."""
    protected_terms = build_protected_terms(company_name)
    fallback = most_common_answer(historical_answers, protected_terms=protected_terms)
    if not ANSWER_GENERATOR:
        return fallback

    clean_answers = [
        redact_personal_data(answer, protected_terms=protected_terms)
        for answer in historical_answers
        if redact_personal_data(answer, protected_terms=protected_terms)
    ]
    clean_examples = [redact_personal_data(example, protected_terms=protected_terms) for example in examples if redact_personal_data(example, protected_terms=protected_terms)]
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
    """Genera sugerencias de FAQ para una sola empresa."""
    load_models()
    existing_questions = existing_questions or []

    valid_items = []
    for item in conversations:
        user_text = normalize_text(item.get("user_text", ""))
        if user_text and is_good_faq_candidate(user_text):
            valid_items.append({**item, "user_text": user_text})

    if len(valid_items) < FAQ_MIN_CLUSTER_SIZE:
        return [], None

    user_texts = [item["user_text"] for item in valid_items]

    # Convierte cada pregunta en un vector numerico comparable.
    embeddings = EMBEDDING_MODEL.encode(
        user_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # DBSCAN no fuerza mensajes sueltos a entrar en una FAQ; los marca como ruido.
    model = DBSCAN(eps=FAQ_CLUSTER_EPS, min_samples=FAQ_MIN_CLUSTER_SIZE, metric="cosine")
    labels = model.fit_predict(embeddings)

    suggestions: List[SuggestionResponse] = []
    cluster_groups: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        if label != -1:
            cluster_groups.setdefault(int(label), []).append(index)

    # Para cada cluster, escoge la pregunta mas cercana al centro como representante.
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
        if support < 0.68:
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
        silhouette = round(silhouette_score(clean_embeddings, clean_labels, metric="cosine"), 4)

    return suggestions, silhouette


def build_suggestions(conversations: List[Dict[str, str]]) -> SuggestionSummary:
    """Genera sugerencias de FAQ agrupando preguntas por empresa y similitud semantica."""
    if not conversations:
        raise ValueError("No conversation pairs available for suggestion generation.")

    conversations_by_company: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for item in conversations:
        conversations_by_company[company_key(item)].append(item)

    suggestions: List[SuggestionResponse] = []
    silhouettes = []
    total_examples = 0
    existing_faqs_by_company = load_existing_faqs_by_company()

    for items in conversations_by_company.values():
        current_company = company_key(items[0]) if items else "unknown"
        total_examples += sum(1 for item in items if is_good_faq_candidate(item.get("user_text", "")))
        company_suggestions, company_silhouette = build_company_suggestions(
            items,
            existing_questions=existing_faqs_by_company.get(current_company, []),
        )
        suggestions.extend(company_suggestions)
        if company_silhouette is not None:
            silhouettes.append(company_silhouette)

    if not suggestions:
        raise ValueError("Insufficient user messages to generate FAQ suggestions.")

    silhouette = round(sum(silhouettes) / len(silhouettes), 4) if silhouettes else None

    # Persiste el resumen para que otros endpoints o servicios puedan consultarlo.
    summary = SuggestionSummary(
        company_count=len(conversations_by_company),
        cluster_count=len(suggestions),
        total_examples=total_examples,
        average_cluster_size=round(total_examples / len(suggestions), 2),
        silhouette_score=silhouette,
        suggestions=suggestions,
    )
    save_json(summary.dict(), SUGGESTIONS_PATH)
    return summary


@app.get("/health")
def health() -> dict:
    """Endpoint simple para confirmar que el servicio esta activo."""
    load_models()
    return {
        "status": "ok",
        "service": "suggestion",
        "embedding_model": MODEL_NAME,
        "answer_model": FAQ_LLM_MODEL if ANSWER_GENERATOR else "historical_fallback",
    }


@app.post("/suggest", response_model=SuggestionSummary)
def suggest() -> SuggestionSummary:
    """Endpoint principal: genera y guarda sugerencias nuevas."""
    conversations = load_conversation_pairs()
    return build_suggestions(conversations)


@app.get("/suggestions", response_model=SuggestionSummary)
def get_suggestions() -> SuggestionSummary:
    """Devuelve la ultima tanda de sugerencias generadas."""
    if not SUGGESTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="No suggestions have been generated yet.")
    raw = load_json(SUGGESTIONS_PATH)
    return SuggestionSummary(**raw)


if __name__ == "__main__":
    import uvicorn

    # Levanta el servicio localmente en el puerto 8003.
    uvicorn.run("suggestion_service:app", host="127.0.0.1", port=8003, log_level="info")

