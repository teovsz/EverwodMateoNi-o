from datetime import datetime
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from faq_common import DATA_DIR, load_json, save_json
from faq_models import ValidationRequest, ValidationResponse

# Aplicacion FastAPI del servicio de validacion humana.
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Everwod FAQ Ingestion Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Archivos que contienen las sugerencias generadas y sus revisiones.
SUGGESTIONS_PATH = DATA_DIR / "faq_suggestions.json"
VALIDATIONS_PATH = DATA_DIR / "faq_validations.json"


def load_suggestions() -> List[Dict[str, Any]]:
    """Carga las sugerencias disponibles para revision humana."""
    if not SUGGESTIONS_PATH.exists():
        raise FileNotFoundError("No FAQ suggestions available. Ejecute el servicio de sugerencias primero.")
    return load_json(SUGGESTIONS_PATH).get("suggestions", [])


def load_validations() -> List[Dict[str, Any]]:
    """Carga validaciones guardadas; devuelve lista vacia si aun no hay archivo."""
    if VALIDATIONS_PATH.exists():
        return load_json(VALIDATIONS_PATH)
    return []


def save_validations(validations: List[Dict[str, Any]]) -> None:
    """Persiste todas las validaciones en un archivo JSON."""
    save_json(validations, VALIDATIONS_PATH)


@app.get("/health")
def health() -> dict:
    """Endpoint simple para confirmar que el servicio esta activo."""
    return {"status": "ok", "service": "validation"}


@app.get("/suggestions")
def get_suggestions() -> List[Dict[str, Any]]:
    """Devuelve las sugerencias pendientes o disponibles para validar."""
    return load_suggestions()


@app.get("/validations")
def get_validations() -> List[Dict[str, Any]]:
    """Devuelve el historial de validaciones realizadas."""
    return load_validations()


@app.post("/validate", response_model=ValidationResponse)
def validate(request: ValidationRequest) -> ValidationResponse:
    """Registra la revision de una sugerencia especifica."""
    suggestions = load_suggestions()

    # Verifica que el ID exista para no guardar revisiones huerfanas.
    suggestion_ids = {item["id"] for item in suggestions}
    if request.suggestion_id not in suggestion_ids:
        raise HTTPException(status_code=404, detail="Suggestion ID not found.")

    # Si Postman no envia fecha, se usa la hora actual UTC.
    validations = load_validations()
    reviewed_at = request.reviewed_at or datetime.utcnow()
    entry = {
        "suggestion_id": request.suggestion_id,
        "reviewer": request.reviewer,
        "status": request.status,
        "notes": request.notes,
        "reviewed_at": reviewed_at.isoformat(),
    }
    validations.append(entry)
    save_validations(validations)
    return ValidationResponse(**entry)


if __name__ == "__main__":
    import uvicorn

    # Levanta el servicio localmente en el puerto 8004.
    uvicorn.run("validation_service:app", host="127.0.0.1", port=8004, log_level="info")
