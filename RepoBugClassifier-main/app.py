"""
FastAPI backend for the Repository-Specific Bug Classification System.

Loads two independent specialist pipelines (Machine Learning, JavaScript),
each consisting of a fitted TF-IDF vectorizer + trained XGBoost classifier.
Routing between them is deterministic, based on the `repository` field
supplied by the caller — no domain-detection model is used.
"""

import logging
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("bug-classifier")

# ---------------------------------------------------------------------------
# Model registry: maps a user-facing repository choice to its specific
# vectorizer + model pair. Add new domains here as the project grows.
# ---------------------------------------------------------------------------
MODEL_PATHS = {
    "Machine Learning": {
        "vectorizer": "models/ML_tfidf.pkl",
        "model": "models/ML_xgb_tfidf.pkl",
    },
    "JavaScript": {
        "vectorizer": "models/javascript_tfidf.pkl",
        "model": "models/javascript_xgb_tfidf.pkl",
    },
}

# populated at startup
MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load all four .pkl files once, at process startup — not per-request.
    for repo_name, paths in MODEL_PATHS.items():
        logger.info(f"Loading models for repository: {repo_name}")
        vectorizer = joblib.load(paths["vectorizer"])
        model = joblib.load(paths["model"])
        MODELS[repo_name] = {"vectorizer": vectorizer, "model": model}
    logger.info("All models loaded successfully.")
    yield
    MODELS.clear()


app = FastAPI(
    title="Repository-Specific Bug Classification API",
    description="Predicts whether a GitHub issue is a Bug or Not Bug, "
                 "using a domain-specific specialist model.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    repository: str = Field(..., description='"Machine Learning" or "JavaScript"')
    title: str = Field(..., description="Issue title")
    body: str = Field(default="", description="Issue body/description")


class PredictResponse(BaseModel):
    repository: str
    prediction: str
    confidence: float
    bug_probability: float
    not_bug_probability: float


@app.get("/health")
def health_check():
    return {"status": "ok", "loaded_repositories": list(MODELS.keys())}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if request.repository not in MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown repository '{request.repository}'. "
                f"Valid options: {list(MODELS.keys())}"
            ),
        )

    combined_text = f"{request.title} {request.body}".strip()
    if not combined_text:
        raise HTTPException(status_code=400, detail="Title and body cannot both be empty.")

    cleaned_text = preprocess_text(combined_text)

    pipeline = MODELS[request.repository]
    vectorizer = pipeline["vectorizer"]
    model = pipeline["model"]

    vectorized = vectorizer.transform([cleaned_text])  # transform, never fit_transform
    probabilities = model.predict_proba(vectorized)[0]  # [P(not_bug), P(bug)]

    not_bug_prob = float(probabilities[0])
    bug_prob = float(probabilities[1])
    prediction = "Bug" if bug_prob >= 0.5 else "Not Bug"
    confidence = max(bug_prob, not_bug_prob)

    return PredictResponse(
        repository=request.repository,
        prediction=prediction,
        confidence=round(confidence * 100, 2),
        bug_probability=round(bug_prob * 100, 2),
        not_bug_probability=round(not_bug_prob * 100, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)