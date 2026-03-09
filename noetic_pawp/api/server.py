from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from noetic_pawp.tokenizer import PAWPTokenizer

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except Exception as exc:  # pragma: no cover
    raise ImportError("FastAPI and Pydantic are required. Install with `pip install fastapi uvicorn`.") from exc


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4096)
    language: str = Field(default="pt", min_length=2, max_length=8)


class PredictResponse(BaseModel):
    tokens: list[dict[str, Any]]


@dataclass
class InferenceService:
    tokenizer: PAWPTokenizer

    def predict(self, text: str, language: str) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.tokenizer.encode(text, language=language)]


def create_app(service: InferenceService | None = None) -> FastAPI:
    service = service or InferenceService(tokenizer=PAWPTokenizer())
    app = FastAPI(title="Noetic API", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        try:
            result = service.predict(payload.text, payload.language)
            return PredictResponse(tokens=result)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Unexpected inference error: {exc}") from exc

    return app


app = create_app()
